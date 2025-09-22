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
    lr=1e-3,
    epochs=100,
    save_dir="checkpoints",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    os.makedirs(save_dir, exist_ok=True)

    # ----- Dataset & Loader -----
    # Create full dataset
    full_dataset = EASTDataset(
        img_dir=img_dir,
        map_dir=map_dir,
        size=512,
        training=True
    )
    
    # Split into train / validation (90% / 10%)
    val_ratio = 0.1
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Train size: {train_size},  Val size: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # ----- Model, Loss, Optimizer -----
    model = EAST().to(device)
    criterion = EASTLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate schedule similar to paper
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=27300 // (train_size // batch_size), gamma=0.1)

    # ----- Early stopping variables -----
    best_val = float("inf")
    patience = 10
    trigger = 0

    print(f"Training on {train_size} images and validating on {val_size} images for {epochs} epochs")
    for epoch in range(1, epochs + 1):
        # ----- Training Loop -----
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", ncols=80)
        for imgs, scores, geos, nmaps in pbar:
            imgs   = imgs.to(device)
            scores = scores.to(device)
            geos   = geos.to(device)
            nmaps  = nmaps.to(device)

            optimizer.zero_grad()
            pred_score, pred_geo = model(imgs)
            total_loss, s_loss, g_loss = criterion(pred_score, pred_geo, scores, geos, nmaps)
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            pbar.set_postfix({
                "total": f"{total_loss.item():.4f}",
                "score": f"{s_loss.item():.4f}",
                "geo":   f"{g_loss.item():.4f}"
            })

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        # ----- Validation Loop -----
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
                total_loss, _, _ = criterion(pred_score, pred_geo, scores, geos, nmaps)
                val_loss += total_loss.item()
        
        val_loss /= len(val_loader)
        print(f"Epoch {epoch} | train loss {avg_loss:.4f} | val loss {val_loss:.4f}")

        # ----- Early Stopping check -----
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            trigger = 0
            # Save the best checkpoint
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, os.path.join(save_dir, "east_best.pth"))
            print(f"✅ New best model saved with val loss: {best_val:.4f}")
        else:
            trigger += 1
            if trigger >= patience:
                print(f"🛑 Early stopping at epoch {epoch}. No improvement for {patience} epochs.")
                break

if __name__ == "__main__":
    train()