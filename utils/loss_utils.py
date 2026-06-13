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
import torch.nn as nn
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
def _tensor_size(t):
    return t.size()[1]*t.size()[2]*t.size()[3]

def tv_loss(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:,:,1:,:])
    count_w = _tensor_size(x[:,:,:,1:])
    h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
    w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
    return 2*(h_tv/count_h+w_tv/count_w)/batch_size

class FrequencyLoss(nn.Module):
    def __init__(self, alpha=1.0, threshold_ratio=0.2,enhance_high=1.0):
        super().__init__()
        self.alpha = alpha
        self.threshold_ratio = threshold_ratio
        self.device = 'cuda'
        self.enhance_high = enhance_high

    def get_high_pass(self, img_tensor):
        # 2. 对每个通道进行傅里叶变换
        fft_channels = []
        for c in range(img_tensor.shape[1]):
            channel_data = img_tensor[:, c, ...]
            fft = torch.fft.fft2(channel_data)
            fft_shifted = torch.fft.fftshift(fft)
            fft_channels.append(fft_shifted)

        fft_complex = torch.stack(fft_channels, dim=1)  # [1, 3, H, W]
        # 3. 创建频率掩模
        h, w = img_tensor.shape[-2], img_tensor.shape[-1]
        cy, cx = h // 2, w // 2
        cutoff = int(min(h, w) * self.threshold_ratio)
        # 创建网格坐标
        y = torch.arange(h, device=img_tensor.device).float() - cy
        x = torch.arange(w, device=img_tensor.device).float() - cx
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        distance = torch.sqrt(yy ** 2 + xx ** 2)
        # 高斯过渡带掩模（避免振铃效应）
        sigma = cutoff / 3
        mask = 1 - torch.exp(-distance ** 2 / (2 * sigma ** 2))
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        # 4. 分解频率成分
        low_freq_fft = fft_complex * (1 - mask)
        high_freq_fft = fft_complex * mask
        # 5. 增强高频成分
        enhanced_high_freq_fft = high_freq_fft * self.enhance_high

        enhanced_fft = (1-self.alpha) * low_freq_fft + self.alpha * enhanced_high_freq_fft
        # 逆变换得到高频信息
        high_freq = torch.zeros_like(img_tensor)
        for c in range(high_freq_fft.shape[1]):
            channel_fft = torch.fft.ifftshift(enhanced_fft[:, c, ...])
            channel_ifft = torch.fft.ifft2(channel_fft).real
            high_freq[:, c, ...] = channel_ifft
        # 标准化高频信息
        high_freq_normalized = (high_freq - high_freq.min()) / (high_freq.max() - high_freq.min())
        return high_freq_normalized


    def forward(self, pred, target):
        batch_size = len(pred)
        total_loss = 0
        for k in range(batch_size):
            pred_high = self.get_high_pass(pred[k].unsqueeze(0))
            target_high = self.get_high_pass(target[k].unsqueeze(0))
            loss_high = F.mse_loss(pred_high, target_high, reduction='mean')
            total_loss+=loss_high
        # 损失计算
        # loss_high = F.l1_loss(pred_high, target_high)
        # loss_high = F.mse_loss(pred_high, target_high, reduction='mean')
        return total_loss/batch_size

