import torch
import torch.nn.functional as F
from torch import nn

EPS = 1e-6

class EASTLoss(nn.Module):
    def __init__(self, lambda_geo=1.0):
        super(EASTLoss, self).__init__()
        self.lambda_geo = lambda_geo

    def forward(self, pred_score, pred_geo, gt_score, gt_geo, gt_nmap):
        """
        pred_score: [B,1,H,W] (sigmoid already applied by model)
        pred_geo  : [B,8,H,W]
        gt_score  : [B,1,H,W] (0/1)
        gt_geo    : [B,8,H,W]
        gt_nmap   : [B,1,H,W] normalization scalar per pixel (0 outside)
        """
        # ---- Score loss (class-balanced cross-entropy) ----
        pos_mask = (gt_score == 1).float()
        neg_mask = (gt_score == 0).float()

        n_pos = pos_mask.sum()
        n_neg = neg_mask.sum()
        # beta as in paper: 1 - (#pos / total) -> equal to n_neg / (n_pos + n_neg)
        beta = n_neg / (n_pos + n_neg + EPS)

        # Weighted bce per pixel
        bce_map = - beta * pos_mask * torch.log(pred_score + EPS) \
                  - (1.0 - beta) * neg_mask * torch.log(1.0 - pred_score + EPS)
        score_loss = bce_map.mean()

        # ---- Geometry loss (scale-normalized smooth L1) ----
        # only compute where gt is positive
        if n_pos.item() < 1:
            geo_loss = torch.tensor(0.).to(pred_score.device)
        else:
            # expand nmap to 8 channels for division
            nmap = gt_nmap.clamp(min=EPS)  # [B,1,H,W]
            nmap8 = nmap.repeat(1, pred_geo.shape[1], 1, 1)  # [B,8,H,W]

            # normalize ground truth and prediction by nmap
            pred_norm = pred_geo / nmap8
            gt_norm   = gt_geo / nmap8

            # mask to compute loss only on positive pixels
            pos8 = pos_mask.repeat(1, pred_geo.shape[1], 1, 1)  # [B,8,H,W]

            # use smooth_l1 (Huber) but compute only where pos8 is 1
            diff = pred_norm - gt_norm
            diff = diff * pos8  # zero out negatives

            # compute smooth L1: sum over elements then divide by #elements (mean on positive elements)
            # Using reduction='sum' then normalize avoids counting zeros from negative pixels.
            loss_sum = F.smooth_l1_loss(pred_norm * pos8, gt_norm * pos8, reduction='sum')
            denom = pos8.sum() + EPS
            geo_loss = loss_sum / denom

        total_loss = score_loss + self.lambda_geo * geo_loss
        return total_loss, score_loss, geo_loss
