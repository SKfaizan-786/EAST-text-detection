import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.dataset import EASTDataset
from src.model import EAST
from src.losses import EASTLoss


def train(
    img_dir="data/icdar2015/train_images",
    map_dir="data/icdar2015/train_maps",
    batch_size=4,
    lr=5e-4,                     # 🔑 lower starting LR
    epochs=100,
    save_dir="checkpoints",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    os.makedirs(save_dir, exist_ok=True)

    # ----- Dataset & Split -----
    full_dataset = EASTDataset(img_dir=img_dir,
                               map_dir=map_dir,
                               size=512,
                               training=True)

    val_ratio = 0.1
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(42)  # reproducible split
    train_dataset, val_dataset = random_split(full_dataset,
                                              [train_size, val_size],
                                              generator=generator)

    print(f"Train size: {train_size},  Val size: {val_size}")

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True)

    # ----- Model / Loss / Optimizer -----
    model = EAST().to(device)
    criterion = EASTLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 🔑 Scheduler: reduce LR when val loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # ----- Early Stopping -----
    best_val = float("inf")
    patience = 20      # 🔑 give more room before stopping
    trigger = 0

    print(f"Training on {train_size} images | validating on {val_size} images")
    print(f"Device: {device}")

    for epoch in range(1, epochs + 1):
        # ===== Training =====
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", ncols=80)
        for imgs, scores, geos, nmaps in pbar:
            imgs, scores, geos, nmaps = (
                imgs.to(device),
                scores.to(device),
                geos.to(device),
                nmaps.to(device),
            )

            optimizer.zero_grad()
            pred_score, pred_geo = model(imgs)
            total_loss, s_loss, g_loss = criterion(
                pred_score, pred_geo, scores, geos, nmaps
            )
            total_loss.backward()

            # 🔑 Gradient clipping to prevent spikes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            epoch_loss += total_loss.item()

            pbar.set_postfix({
                "total": f"{total_loss.item():.4f}",
                "score": f"{s_loss.item():.4f}",
                "geo":   f"{g_loss.item():.4f}"
            })

        avg_train_loss = epoch_loss / len(train_loader)

        # ===== Validation =====
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, scores, geos, nmaps in val_loader:
                imgs, scores, geos, nmaps = (
                    imgs.to(device),
                    scores.to(device),
                    geos.to(device),
                    nmaps.to(device),
                )
                pred_score, pred_geo = model(imgs)
                loss, _, _ = criterion(pred_score, pred_geo,
                                       scores, geos, nmaps)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch} | train {avg_train_loss:.4f} | val {val_loss:.4f}")

        # 🔑 Step the scheduler with validation loss
        scheduler.step(val_loss)

        # ===== Early Stopping =====
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            trigger = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, os.path.join(save_dir, "east_best.pth"))
            print(f"✅ New best model saved with val loss {best_val:.4f}")
        else:
            trigger += 1
            if trigger >= patience:
                print(f"🛑 Early stopping at epoch {epoch} "
                      f"(no val improvement for {patience} epochs).")
                break


if __name__ == "__main__":
    train()
