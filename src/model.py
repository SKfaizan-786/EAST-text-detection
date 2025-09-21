import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class EAST(nn.Module):
    def __init__(self, pretrained=True):
        super(EAST, self).__init__()

        # -------- Backbone: ResNet-18 --------
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Extract layers up to each stage
        self.stage1 = nn.Sequential(resnet.conv1, resnet.bn1,
                                    resnet.relu, resnet.maxpool,
                                    resnet.layer1)  # 1/4 size
        self.stage2 = resnet.layer2  # 1/8
        self.stage3 = resnet.layer3  # 1/16
        self.stage4 = resnet.layer4  # 1/32

        # -------- Feature merging branch --------
        # Reduce channels for merging
        self.reduce4 = nn.Conv2d(512, 128, 1)
        self.reduce3 = nn.Conv2d(256, 128, 1)
        self.reduce2 = nn.Conv2d(128, 128, 1)
        self.reduce1 = nn.Conv2d(64,  128, 1)

        self.merge3 = nn.Conv2d(128, 128, 3, padding=1)
        self.merge2 = nn.Conv2d(128, 128, 3, padding=1)
        self.merge1 = nn.Conv2d(128, 128, 3, padding=1)

        # -------- Output layers --------
        self.score_head = nn.Conv2d(128, 1, 1)   # 1-channel score map
        self.geo_head   = nn.Conv2d(128, 8, 1)   # 8-channel geometry map

    def forward(self, x):
        # Backbone forward
        c1 = self.stage1(x)  # 1/4
        c2 = self.stage2(c1) # 1/8
        c3 = self.stage3(c2) # 1/16
        c4 = self.stage4(c3) # 1/32

        # Top-down merging
        h4 = self.reduce4(c4)

        h3 = self._upsample_add(h4, self.reduce3(c3))
        h3 = F.relu(self.merge3(h3))

        h2 = self._upsample_add(h3, self.reduce2(c2))
        h2 = F.relu(self.merge2(h2))

        h1 = self._upsample_add(h2, self.reduce1(c1))
        h1 = F.relu(self.merge1(h1))

        # Upsample to full resolution
        h = F.interpolate(h1, scale_factor=4, mode='bilinear', align_corners=True)

        score = torch.sigmoid(self.score_head(h))
        geo   = self.geo_head(h)  # raw offsets

        return score, geo

    @staticmethod
    def _upsample_add(x, y):
        """Upsample x to y's size and add."""
        return F.interpolate(x, size=y.shape[2:], mode='bilinear', align_corners=True) + y
