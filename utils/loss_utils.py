#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torchsort 
import math
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()

def spearman_depth_loss_zscore(depth_src, depth_target, sample_size=2048, regularization_strength=1.0):
    # (B, H, W) → (B, H*W)
    depth_src = depth_src.reshape(depth_src.size(0), -1)
    depth_target = depth_target.reshape(depth_target.size(0), -1)

    B, num_points = depth_src.shape
    sample_size = min(sample_size, num_points)

    # (B, sample_size) 만큼 인덱스 선택
    idx = torch.randint(0, num_points, (B, sample_size), device=depth_src.device)

    # 샘플링
    src_sampled = depth_src.gather(dim=1, index=idx)
    tgt_sampled = depth_target.gather(dim=1, index=idx)

    # ⭐ Z-score 정규화
    src_z = (src_sampled - src_sampled.mean(dim=1, keepdim=True)) / (src_sampled.std(dim=1, keepdim=True) + 1e-6)
    tgt_z = (tgt_sampled - tgt_sampled.mean(dim=1, keepdim=True)) / (tgt_sampled.std(dim=1, keepdim=True) + 1e-6)

    # 동일한 tau 사용
    src_rank = torchsort.soft_rank(src_z, regularization_strength=regularization_strength)
    tgt_rank = torchsort.soft_rank(tgt_z, regularization_strength=regularization_strength)

    # 평균 제거 및 L2 정규화
    src_rank = src_rank - src_rank.mean(dim=1, keepdim=True)
    tgt_rank = tgt_rank - tgt_rank.mean(dim=1, keepdim=True)

    src_rank = src_rank / (src_rank.norm(dim=1, keepdim=True) + 1e-6)
    tgt_rank = tgt_rank / (tgt_rank.norm(dim=1, keepdim=True) + 1e-6)

    # 배치별 Spearman correlation
    corr = (src_rank * tgt_rank).sum(dim=1)

    assert not torch.any(torch.isnan(corr)), "NaN in Spearman correlation!"

    loss = 1 - corr  # Spearman similarity → loss
    return loss.mean()