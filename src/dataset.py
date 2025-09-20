import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from glob import glob

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class EASTDataset(Dataset):
    def __init__(self, img_dir, map_dir, size=512, training=True,
                 positive_crop_prob=0.5, normalize=True):
        """
        img_dir : images folder
        map_dir : folder with <name>_score.npy and <name>_geo.npy
        size    : output crop size (512 in paper)
        training: whether to apply random crops/aug
        positive_crop_prob: prob of sampling a crop that contains text
        normalize: apply ImageNet mean/std normalization
        """
        self.img_paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
        self.map_dir = map_dir
        self.size = size
        self.training = training
        self.pos_prob = positive_crop_prob
        self.normalize = normalize

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        name = os.path.splitext(os.path.basename(img_path))[0]

        img = cv2.imread(img_path)[:, :, ::-1].astype(np.float32) / 255.0  # RGB float32 [0,1]
        score = np.load(os.path.join(self.map_dir, f"{name}_score.npy"))   # HxW, uint8 or 0/1
        geo   = np.load(os.path.join(self.map_dir, f"{name}_geo.npy"))     # HxW x 8, float32 offsets

        # ensure types
        score = score.astype(np.uint8)
        geo = geo.astype(np.float32)

        if self.training:
            img, score, geo = self._preprocess_train(img, score, geo)
        else:
            img, score, geo = self._preprocess_val(img, score, geo)

        # normalization (ImageNet) if requested
        if self.normalize:
            img = (img - IMAGENET_MEAN.reshape(1,1,3)) / IMAGENET_STD.reshape(1,1,3)

        # HWC -> CHW and to torch
        img = torch.from_numpy(img.transpose(2,0,1)).float()
        score = torch.from_numpy(score.astype(np.float32)).unsqueeze(0).float()
        geo   = torch.from_numpy(geo.transpose(2,0,1)).float()

        return img, score, geo

    def _preprocess_train(self, img, score, geo):
        h, w = img.shape[:2]

        # if too small, scale up so min(h,w) >= size (keep aspect)
        if min(h, w) < self.size:
            scale = float(self.size) / float(min(h, w))
            img, score, geo = self._resize(img, score, geo, scale)
            h, w = img.shape[:2]

        # decide positive or random crop
        if random.random() < self.pos_prob and np.any(score > 0):
            img, score, geo = self._random_crop_with_text(img, score, geo, self.size)
        else:
            img, score, geo = self._random_crop(img, score, geo, self.size)

        # horizontal flip (and adjust geo x offsets)
        if random.random() < 0.5:
            img = img[:, ::-1, :].copy()
            score = score[:, ::-1].copy()
            geo = geo[:, ::-1, :].copy()
            # negate x-offset channels (dx) after horizontal flip: channels 0,2,4,6
            geo[..., 0::2] = -geo[..., 0::2]

        return img, score, geo

    def _preprocess_val(self, img, score, geo):
        # simple center crop or scale to target size
        h, w = img.shape[:2]
        if min(h, w) < self.size:
            scale = float(self.size) / float(min(h, w))
            img, score, geo = self._resize(img, score, geo, scale)
            h, w = img.shape[:2]
        # center crop
        x = max(0, (w - self.size) // 2)
        y = max(0, (h - self.size) // 2)
        img = img[y:y+self.size, x:x+self.size]
        score = score[y:y+self.size, x:x+self.size]
        geo = geo[y:y+self.size, x:x+self.size]
        return img, score, geo

    def _resize(self, img, score, geo, scale):
        """ Resize image, score and geo with correct interpolation and scaling of offsets. """
        h, w = img.shape[:2]
        new_w = int(w * scale + 0.5)
        new_h = int(h * scale + 0.5)

        img_rs = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        score_rs = cv2.resize(score.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        geo_rs = np.zeros((new_h, new_w, 8), dtype=np.float32)
        sx = float(new_w) / float(w)
        sy = float(new_h) / float(h)

        # resize each offset channel and scale values
        for k in range(4):
            dx = geo[..., 2*k]
            dy = geo[..., 2*k+1]
            dx_rs = cv2.resize(dx, (new_w, new_h), interpolation=cv2.INTER_LINEAR) * sx
            dy_rs = cv2.resize(dy, (new_w, new_h), interpolation=cv2.INTER_LINEAR) * sy
            geo_rs[..., 2*k]   = dx_rs
            geo_rs[..., 2*k+1] = dy_rs

        return img_rs, score_rs, geo_rs

    def _random_crop(self, img, score, geo, size):
        h, w = img.shape[:2]
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        return (img[y:y+size, x:x+size],
                score[y:y+size, x:x+size],
                geo[y:y+size, x:x+size])

    def _random_crop_with_text(self, img, score, geo, size):
        """ Sample a crop that contains at least one positive pixel.
            Strategy: pick a random positive pixel and center crop around it with jitter.
        """
        h, w = img.shape[:2]
        ys, xs = np.where(score > 0)
        if len(xs) == 0:
            return self._random_crop(img, score, geo, size)

        # pick a random positive pixel
        idx = random.randint(0, len(xs)-1)
        cx = xs[idx]
        cy = ys[idx]

        # jitter center a bit so crop is not always centered
        jitter = size // 4
        cx = np.clip(cx + random.randint(-jitter, jitter), 0, w-1)
        cy = np.clip(cy + random.randint(-jitter, jitter), 0, h-1)

        x1 = np.clip(cx - size//2, 0, w - size)
        y1 = np.clip(cy - size//2, 0, h - size)

        return (img[y1:y1+size, x1:x1+size],
                score[y1:y1+size, x1:x1+size],
                geo[y1:y1+size, x1:x1+size])
