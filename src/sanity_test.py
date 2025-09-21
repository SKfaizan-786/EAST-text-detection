import torch
from torch.utils.data import DataLoader
from src.dataset import EASTDataset
from src.model import EAST
from src.losses import EASTLoss
import torch.optim as optim
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Tiny dataset / loader ---
train_ds = EASTDataset(
    img_dir="data/icdar2015/train_images",
    map_dir="data/icdar2015/train_maps",
    size=512,
    training=True
)
# use a very small subset to make it fast
subset_size = 16
train_ds.img_paths = train_ds.img_paths[:subset_size]

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)

# --- Model / loss / optimizer ---
model = EAST().to(device)
criterion = EASTLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# --- One quick epoch ---
model.train()
for epoch in range(1, 2):   # 1 epoch
    epoch_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/1", ncols=80)
    for imgs, scores, geos, nmaps in pbar:
        imgs, scores, geos, nmaps = (
            imgs.to(device),
            scores.to(device),
            geos.to(device),
            nmaps.to(device),
        )

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

    print(f"Epoch {epoch} average loss: {epoch_loss/len(train_loader):.4f}")
