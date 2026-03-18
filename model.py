import math
import torch
import torch.nn as nn
from torchvision import models


class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # ResNet-18 backbone
        self.backbone = models.resnet18(weights="IMAGENET1K_V1")
        self.backbone.fc = nn.Identity()  # remove FC after GAP

    def forward(self, x):
        return self.backbone(x)  # Output: (B, 512)


class StateEncoder(nn.Module):
    def __init__(self, input_dim=2, output_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256), nn.Mish(), nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)  # Output: (B, 64)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        # set frequency -> exp(i * -ln(10000) / (D/2 - 1))
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        # \theta_{k,i} = k * \omega_i
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionCondition(nn.Module):
    def __init__(self):
        super().__init__()
        self.vis_enc = VisualEncoder()
        self.state_enc = StateEncoder()
        self.time_enc = nn.Sequential(
            SinusoidalPosEmb(64), nn.Linear(64, 128), nn.Mish(), nn.Linear(128, 64)
        )
        # final size: 512(Vis) + 64(State) + 64(Time) = 640

    def forward(self, image, state, k):
        v_emb = self.vis_enc(image)
        s_emb = self.state_enc(state)
        t_emb = self.time_enc(k)

        return torch.cat([v_emb, s_emb, t_emb], dim=-1)


class FiLMBlock1d(nn.Module):
    def __init__(self, cond_dim, out_channels):
        super().__init__()
        # Generate gamma, beta from condition (C)

        self.generator = nn.Linear(cond_dim, out_channels * 2)

    def forward(self, x, condition):
        # Generate gamma and beta
        # x shape: (Batch, Channels, Horizon)
        # condition shape: (Batch, cond_dim)
        params = self.generator(condition)  # (Batch, 2 * out_channels)
        params = params.unsqueeze(-1)  # (Batch, 2 * out_channels, 1)

        gamma, beta = params.chunk(2, dim=1)  # (Batch, out_channels, 1) x 2

        # Affine Transformation
        return x * gamma + beta


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.film = FiLMBlock1d(cond_dim, out_channels)
        self.activation = nn.Mish()

    def forward(self, x, condition):
        x = self.conv(x)
        x = self.film(x, condition)
        return self.activation(x)


class DownsampleBlock(nn.Module):
    def __init__(self, in_c, out_c, cond_dim):
        super().__init__()
        # half in H
        self.conv = Conv1dBlock(in_c, out_c, cond_dim)
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x, cond):
        return self.pool(self.conv(x, cond))


class UpsampleBlock(nn.Module):
    def __init__(self, in_c, out_c, cond_dim):
        super().__init__()
        # double in H
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.conv = Conv1dBlock(in_c, out_c, cond_dim)

    def forward(self, x, skip, cond):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x, cond)


class ConditionalTemporalUnet1d(nn.Module):
    def __init__(self, action_dim, cond_dim=640):
        super().__init__()
        self.cond_enc = DiffusionCondition()

        # Encoder
        self.inc = Conv1dBlock(action_dim, 64, cond_dim)  # x1: (H=16, C=64)
        self.down1 = DownsampleBlock(64, 128, cond_dim)  # x2: (H=8,  C=128)
        self.down2 = DownsampleBlock(128, 256, cond_dim)  # x3: (H=4,  C=256)
        self.down3 = DownsampleBlock(256, 512, cond_dim)  # x4: (H=2,  C=512)

        # Bottleneck
        self.mid = Conv1dBlock(512, 512, cond_dim)  # mid: (H=2, C=512)

        # Decoder
        self.up1 = UpsampleBlock(512 + 256, 256, cond_dim)
        self.up2 = UpsampleBlock(256 + 128, 128, cond_dim)
        self.up3 = UpsampleBlock(128 + 64, 64, cond_dim)

        self.outc = nn.Conv1d(64, action_dim, kernel_size=1)

    def forward(self, action_k, timestep, obs_dict):
        x = action_k.transpose(1, 2)  # (32, 16, 2) -> (32, 2, 16)
        cond = self.cond_enc(obs_dict["image"], obs_dict["agent_pos"], timestep)

        # Encoder Path
        x1 = self.inc(x, cond)  # (32, 64, 16)
        x2 = self.down1(x1, cond)  # (32, 128, 16) -> (32, 128, 8)
        x3 = self.down2(x2, cond)  # (32, 256, 8) -> (32, 256, 4)
        x4 = self.down3(x3, cond)  # (32, 512, 4) -> (32, 512, 2)

        # Bottleneck
        x = self.mid(x4, cond)

        # Decoder
        x = self.up1(x, x3, cond)
        x = self.up2(x, x2, cond)
        x = self.up3(x, x1, cond)

        out = self.outc(x)
        return out.transpose(1, 2)
