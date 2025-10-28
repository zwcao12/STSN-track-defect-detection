"""Torch model for student, teacher and autoencoder model in STSN"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import random
from enum import Enum

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision import transforms
from einops.layers.torch import Rearrange
logger = logging.getLogger(__name__)


def imagenet_norm_batch(x):
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].to(x.device)
    x_norm = (x - mean) / std
    return x_norm


def reduce_tensor_elems(tensor: torch.Tensor, m=2**24) -> torch.Tensor:
    """Flattens n-dimensional tensors,  selects m elements from it
    and returns the selected elements as tensor. It is used to select
    at most 2**24 for torch.quantile operation, as it is the maximum
    supported number of elements.
    https://github.com/pytorch/pytorch/blob/b9f81a483a7879cd3709fd26bcec5f1ee33577e6/aten/src/ATen/native/Sorting.cpp#L291

    Args:
        tensor (torch.Tensor): input tensor from which elements are selected
        m (int): number of maximum tensor elements. Default: 2**24

    Returns:
            Tensor: reduced tensor
    """
    tensor = torch.flatten(tensor)
    if len(tensor) > m:
        # select a random subset with m elements.
        perm = torch.randperm(len(tensor), device=tensor.device)
        idx = perm[:m]
        tensor = tensor[idx]
    return tensor


class STSNModelSize(str, Enum):
    """Supported STSN model sizes"""

    S = "small"


class PDN_Tea_S(nn.Module):
    """Patch Description Network small

    Args:
        out_channels (int): number of convolution output channels
    """

    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__()
        pad_mult = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv4 = nn.Conv2d(256, out_channels, kernel_size=4, stride=1, padding=0 * pad_mult)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)

    def forward(self, x1, x2):
        x1 = imagenet_norm_batch(x1)
        x2 = imagenet_norm_batch(x2)
        x = 0.5 * x1 + 0.5 * x2
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x

class PDN_StuD_S(nn.Module):
    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__()
        pad_mult = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv4 = nn.Conv2d(256, out_channels, kernel_size=4, stride=1, padding=0 * pad_mult)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)

    def forward(self, x):
        x = imagenet_norm_batch(x)
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x

class PDN_StuRGB_S(nn.Module):
    """Patch Description Network small

    Args:
        out_channels (int): number of convolution output channels
    """

    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__()
        pad_mult = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv4 = nn.Conv2d(256, out_channels, kernel_size=4, stride=1, padding=0 * pad_mult)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)

    def forward(self, x):
        x = imagenet_norm_batch(x)
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn

class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn

class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        # pattn2 = self.pa2(x2)
        # pattn2 = self.sigmoid(pattn2)
        # return pattn2
        return x2
class TFAA_Block(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel):
        super(CCMA_Block, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channel, hidden_channel, kernel_size=1, stride=1, padding=0
        )
        self.conv2 = nn.Conv2d(
            in_channel, hidden_channel, kernel_size=1, stride=1, padding=0
        )
        self.conv3 = nn.Conv2d(
            in_channel, hidden_channel, kernel_size=1, stride=1, padding=0
        )
        self.conv5 = nn.Conv2d(
            in_channel, hidden_channel, kernel_size=1, stride=1, padding=0
        )
        self.scale = hidden_channel ** -0.5

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                hidden_channel, out_channel, kernel_size=1, stride=1, padding=0
            ),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.ca = ChannelAttention(768)
        self.sa = SpatialAttention()
        self.pa = PixelAttention(768)
        
    def forward(self, rgb, depth):
        _, _, h, w = rgb.size()

        q = self.conv1(rgb)
        k = self.conv2(depth)
        v = self.conv3(depth)
        p = self.conv5(rgb)
        q = q.view(q.size(0), q.size(1), q.size(2) * q.size(3)).transpose(
            -2, -1
        )
        
        k = k.view(k.size(0), k.size(1), k.size(2) * k.size(3))

        attn = torch.matmul(q, k) * self.scale
        m = attn.softmax(dim=-1)  # [1, 50000, 50000]

        v = v.view(v.size(0), v.size(1), v.size(2) * v.size(3)).transpose(
            -2, -1
        )
       
        z = torch.matmul(m, v)* self.scale
        
        n = z.softmax(dim=-1)
        
        p = p.view(p.size(0), p.size(1), p.size(2) * p.size(3))
        g = torch.matmul(n, p)
        
        g = g.view(g.size(0), h, w, -1)
        g = g.permute(0, 3, 1, 2).contiguous()
        
        output = self.conv4(g)
        # print(output.shape)
        initial = rgb + depth
        initial2 = self.pa(rgb, depth)
        cattan = self.ca(initial)
        # print(cattan.shape)
        sattan = self.sa(initial2)
        # print(sattan.shape)
        output = output + cattan + sattan
        # print(output.shape)
        return output

class Encoder(nn.Module):
    """Autoencoder Encoder model."""

    def __init__(self) -> None:
        super().__init__()
        self.enconv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.enconv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.enconv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv6 = nn.Conv2d(64, 64, kernel_size=8, stride=1, padding=0)

    def forward(self, x):
        x = F.relu(self.enconv1(x))
        x = F.relu(self.enconv2(x))
        x = F.relu(self.enconv3(x))
        x = F.relu(self.enconv4(x))
        x = F.relu(self.enconv5(x))
        x = self.enconv6(x)
        return x


class Decoder(nn.Module):
    """Autoencoder Decoder model.

    Args:
        out_channels (int): number of convolution output channels
    """

    def __init__(self, out_channels, padding, img_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.img_size = img_size
        self.last_upsample = int(img_size / 4) if padding else int(img_size / 4) - 8
        self.deconv1 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv2 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv6 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv8 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.2)
        self.dropout5 = nn.Dropout(p=0.2)
        self.dropout6 = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.interpolate(x, size=int(self.img_size / 64) - 1, mode="bilinear")
        x = F.relu(self.deconv1(x))
        x = self.dropout1(x)
        x = F.interpolate(x, size=int(self.img_size / 32), mode="bilinear")
        x = F.relu(self.deconv2(x))
        x = self.dropout2(x)
        x = F.interpolate(x, size=int(self.img_size / 16) - 1, mode="bilinear")
        x = F.relu(self.deconv3(x))
        x = self.dropout3(x)
        x = F.interpolate(x, size=int(self.img_size / 8), mode="bilinear")
        x = F.relu(self.deconv4(x))
        x = self.dropout4(x)
        x = F.interpolate(x, size=int(self.img_size / 4) - 1, mode="bilinear")
        x = F.relu(self.deconv5(x))
        x = self.dropout5(x)
        x = F.interpolate(x, size=int(self.img_size / 2) - 1, mode="bilinear")
        x = F.relu(self.deconv6(x))
        x = self.dropout6(x)
        x = F.interpolate(x, size=self.last_upsample, mode="bilinear")
        x = F.relu(self.deconv7(x))
        x = self.deconv8(x)
        return x

class STSNModel(nn.Module):
    """STSN model.

    Args:
        teacher_out_channels (int): number of convolution output channels of the pre-trained teacher model
        pretrained_models_dir (str): path to the pretrained model weights
        input_size (tuple): size of input images
        model_size (str): size of student and teacher model
        padding (bool): use padding in convoluional layers
        pad_maps (bool): relevant if padding is set to False. In this case, pad_maps = True pads the
            output anomaly maps so that their size matches the size in the padding = True case.
        device (str): which device the model should be loaded on
    """

    def __init__(
        self,
        teacher_out_channels: int,
        input_size: tuple[int, int],
        model_size: STSNMyModelSize = STSNMyModelSize.S,
        padding=False,
        pad_maps=True,
    ) -> None:
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pad_maps = pad_maps
        self.teacher:  PDN_Tea_S
        self.student_rgb: PDN_StuRGB_S
        self.student_d:  PDN_StuD_S
        self.tfaa: TFAA_Block
        self.tfaa = TFAA_Block(768, 3136, 768) #3136 2304
        if model_size == STSNMyModelSize.M:
            self.teacher = PDN_M(out_channels=teacher_out_channels, padding=padding).eval()
            self.student = PDN_M(out_channels=teacher_out_channels * 2, padding=padding)

        elif model_size == STSNMyModelSize.S:
            self.teacher = PDN_Tea_S(out_channels=teacher_out_channels, padding=padding).eval()
            self.student_rgb = PDN_StuRGB_S(out_channels=teacher_out_channels * 2, padding=padding)
            self.student_d = PDN_StuD_S(out_channels=teacher_out_channels * 2, padding=padding)
        else:
            raise ValueError(f"Unknown model size {model_size}")

        self.teacher_out_channels: int = teacher_out_channels
        self.input_size: tuple[int, int] = input_size

        self.mean_std: nn.ParameterDict = nn.ParameterDict(
            {
                "mean": torch.zeros((1, self.teacher_out_channels, 1, 1)),
                "std": torch.zeros((1, self.teacher_out_channels, 1, 1)),
            }
        )

        self.quantiles: nn.ParameterDict = nn.ParameterDict(
            {
                "qa_st_rgb": torch.tensor(0.0),
                "qb_st_rgb": torch.tensor(0.0),
                "qa_st_d": torch.tensor(0.0),
                "qb_st_d": torch.tensor(0.0),
                
            }
        )

    def is_set(self, p_dic: nn.ParameterDict) -> bool:
        for _, value in p_dic.items():
            if value.sum() != 0:
                return True
        return False

    def choose_random_aug_image(self, image: Tensor) -> Tensor:
        transform_functions = [
            transforms.functional.adjust_brightness,
            transforms.functional.adjust_contrast,
            transforms.functional.adjust_saturation,
        ]
        # Sample an augmentation coefficient λ from the uniform distribution U(0.8, 1.2)
        coefficient = random.uniform(0.8, 1.2)  # nosec: B311
        transform_function = random.choice(transform_functions)  # nosec: B311
        return transform_function(image, coefficient)

    def forward(self, batch: Tensor, batch_depth: Tensor, batch_imagenet: Tensor = None) -> Tensor | dict:
        """Prediction by STSN models.

        Args:
            batch (Tensor): Input images.

        Returns:
            Tensor: Predictions
        """
        with torch.no_grad():
            teacher_out = self.teacher(batch, batch_depth)
            if self.is_set(self.mean_std):
                """teacher RGBD 1 output"""
                teacher_out = (teacher_out - self.mean_std["mean"]) / self.mean_std["std"]
                # print(teacher_out.shape)
        """student RGB 1 output"""
        student_out = self.student_rgb(batch)
        # print(student_out.shape)
        """student D 1 output"""
        student_out_d = self.student_d(batch_depth)
        # print(student_out_d.shape)
        """teacher and studentRGB"""
        distance_st_rgb = torch.pow(teacher_out - student_out[:, : self.teacher_out_channels, :, :], 2)
       
        """teacher and studentD"""
        distance_st_d = torch.pow(teacher_out - student_out_d[:, : self.teacher_out_channels, :, :], 2)
        """teacher and student RGB and student D"""
        student_out_rgbd = self.tfaa(student_out, student_out_d)
        # print(student_out_rgbd.shape)
        distance_st_rgbd = torch.pow(teacher_out - student_out_rgbd[:, : self.teacher_out_channels, :, :], 2)

        if self.training:
            """Teacher and StudentRGB loss"""
            distance_st_rgb = reduce_tensor_elems(distance_st_rgb)
            d_hard_rgb = torch.quantile(distance_st_rgb, 0.999)
            loss_hard_rgb = torch.mean(distance_st_rgb[distance_st_rgb >= d_hard_rgb])
            student_batch_out = self.student_rgb(batch_imagenet)
            student_output_penalty = student_batch_out[:, : self.teacher_out_channels, :, :]
            loss_penalty_rgb = torch.mean(student_output_penalty**2)# cheng fa xiang
            loss_st_rgb = loss_hard_rgb + loss_penalty_rgb
            
            """Teacher and StudentD loss"""
            distance_st_d = reduce_tensor_elems(distance_st_d)
            d_hard_d = torch.quantile(distance_st_d, 0.999)
            loss_hard_d = torch.mean(distance_st_d[distance_st_d >= d_hard_d])
            student_batch_out_d = self.student_d(batch_imagenet)
            student_output_penalty_d = student_batch_out_d[:, : self.teacher_out_channels, :, :]
            loss_penalty_d = torch.mean(student_output_penalty_d ** 2)  # cheng fa xiang
            loss_st_d = loss_hard_d + loss_penalty_d
            
            """Teacher and StudentRGB and StudentD loss"""
            loss_st_rgbd =  torch.mean(distance_st_rgbd)

            return (loss_st_rgb, loss_st_d, loss_st_rgbd)  # 3 ge biaoliang
        else:
            
            map_st_rgb = torch.mean(distance_st_rgb, dim=1, keepdim=True)
            map_st_d = torch.mean(distance_st_d, dim=1, keepdim=True)
            if self.pad_maps:
                map_st_rgb = F.pad(map_st_rgb, (4, 4, 4, 4))
                map_st_d = F.pad(map_st_d, (4, 4, 4, 4))
            map_st_rgb = F.interpolate(map_st_rgb, size=(self.input_size[0], self.input_size[1]), mode="bilinear")
            map_st_d = F.interpolate(map_st_d, size=(self.input_size[0], self.input_size[1]), mode="bilinear")

            if self.is_set(self.quantiles):
                map_st_rgb = 0.1 * (map_st_rgb - self.quantiles["qa_st_rgb"]) / (self.quantiles["qb_st_rgb"] - self.quantiles["qa_st_rgb"])
                map_st_d = (0.1 * (map_st_d - self.quantiles["qa_st_d"]) / (self.quantiles["qb_st_d"] - self.quantiles["qa_st_d"]))
            map_combined = torch.mul(map_st_rgb, map_st_d)
            # print(map_combined.shape)
            return {"anomaly_map_combined": map_combined, "map_st_rgb": map_st_rgb, "map_st_d": map_st_d}
