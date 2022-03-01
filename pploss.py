import torch
import torch.nn as nn

class PPLoss(nn.Module):
    def __init__(self, dist_bg_fake=15.0, dist_bg_real=2.0, dist_fake_real=17.0):
        """pairwise patch loss

        Args:
            dist_bg_fake (float, optional): distance between background patch and fake patch. Defaults to 15.0.
            dist_bg_real (float, optional): distance between background patch and real patch. Defaults to 2.0.
            dist_fake_real (float, optional): distance between fake patch and real patch. Defaults to 17.0.
        """
        super(PPLoss, self).__init__()
        self.dist_bg_fake = dist_bg_fake
        self.dist_bg_real = dist_bg_real
        self.dist_fake_real = dist_fake_real
        self.pdist = nn.PairwiseDistance(p=2)  # distance loss

    def forward(self, x, patch_pair, patch_gt, patch_pair_weight):
        """calculate pairwise patch loss

        Args:
            x (torch.float64): patch feature
            patch_pair (torch.int64): patch pair for calculate
            patch_gt (torch.int64): patch ground truth
            patch_pair_weight (torch.float64): patch pair score

        Returns:
            loss (torch.float64): result of pairwise patch loss
        """
        b, _, _ = x.shape
        loss = 0.0
        for i in range(b):
            pair_dist = self.pdist(x[i][patch_pair[i][0]], x[i][patch_pair[i][1]])
            gt_dist_bg_fake = (torch.abs(patch_gt[i][patch_pair[i][0]] - patch_gt[i][patch_pair[i][1]]) == 2).float() * self.dist_bg_fake
            gt_dist_bg_real = (torch.abs(patch_gt[i][patch_pair[i][0]] + patch_gt[i][patch_pair[i][1]]) == 1).float() * self.dist_bg_real
            gt_dist_fake_real = (torch.abs(patch_gt[i][patch_pair[i][0]] + patch_gt[i][patch_pair[i][1]]) == 3).float() * self.dist_fake_real
            gt_dist = gt_dist_bg_fake + gt_dist_bg_real + gt_dist_fake_real
            loss += (patch_pair_weight[i] * torch.abs(gt_dist - pair_dist)).mean()
        loss = loss / float(b)
        return loss
