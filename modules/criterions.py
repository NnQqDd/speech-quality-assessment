# reference https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf
# https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/losses/angularloss.py

import torch

class AngularSoftmaxLoss(torch.nn.Module):
    def __init__(self, eps=1e-7, scale=30.0, margin=0.2):
        super().__init__()
        self.eps = eps
        self.scale = scale
        self.margin = margin

    def forward(self, logits, labels):
        numerator = self.scale * torch.cos(
            torch.acos(torch.clamp(torch.diagonal(logits.transpose(0, 1)[labels]), -1.0 + self.eps, 1 - self.eps))
            + self.margin
        )
        excl = torch.cat(
            [torch.cat((logits[i, :y], logits[i, y + 1 :])).unsqueeze(0) for i, y in enumerate(labels)], dim=0
        )
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.scale * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)