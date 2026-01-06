import torch
import torch.nn as nn
import torch.nn.functional as F

class CSCA_Module(nn.Module):
    """
    Cross-Scale Context Attention (CSCA)
    - Captures contextual relationships between multiscale lesion features.
    - Learns how features from different receptive fields influence each other.
    """

    def __init__(self, in_channels, reduction=16):
        super(CSCA_Module, self).__init__()

        # 1x1 conv to reduce dimensionality
        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Learnable scaling parameter
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: [B, C, H, W]
        Returns: enhanced feature map [B, C, H, W]
        """
        B, C, H, W = x.size()

        # Flatten spatial dimensions
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # [B, N, C']
        proj_key   = self.key_conv(x).view(B, -1, H * W)                     # [B, C', N]
        energy = torch.bmm(proj_query, proj_key)                             # [B, N, N]
        attention = self.softmax(energy)                                     # spatial attention map

        proj_value = self.value_conv(x).view(B, -1, H * W)                   # [B, C, N]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))              # [B, C, N]
        out = out.view(B, C, H, W)

        # Residual connection with learnable scaling
        out = self.gamma * out + x
        return out
