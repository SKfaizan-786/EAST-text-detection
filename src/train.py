import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
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
    train_dataset = EASTDataset(
        img_dir=img_dir,
        map_dir=map_dir,
        size=512,
        training=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # ----- Model, Loss, Optimizer -----
    model = EAST().to(device)
    criterion = EASTLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate schedule similar to paper
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=27300 // len(train_loader), gamma=0.1)

    print(f"Training on {len(train_dataset)} images for {epochs} epochs")
    for epoch in range(1, epochs + 1):
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
        print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(save_dir, f"east_epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, ckpt_path)

if __name__ == "__main__":
    train()