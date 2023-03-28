from ..builder import NECKS
import torch
import torch.nn as nn

@NECKS.register_module()
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        # 通道注意力模块
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # 全局平均池化
            nn.Conv2d(in_channels, in_channels // reduction, 1), # 降维
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1), # 升维
            nn.Sigmoid() # 激活函数
        )
        # 空间注意力模块
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3), # 卷积层
            nn.Sigmoid() # 激活函数
        )

    def forward(self, x):
        if isinstance(x, tuple):
            x = list(x)
            for i in range(5):
                # 通道注意力
                channel_weight = self.channel_attention(x[i]) # 计算每个通道的权重
                out = x[i] * channel_weight # 对输入特征进行加权

                # 空间注意力
                max_pool = torch.max(out, dim=1, keepdim=True)[0] # 最大池化
                avg_pool = torch.mean(out, dim=1, keepdim=True) # 平均池化
                spatial_weight = self.spatial_attention(torch.cat([max_pool, avg_pool], dim=1)) # 计算每个位置的权重
                x[i] = out * spatial_weight # 对输入特征进行加权
        else:
            # 通道注意力
            channel_weight = self.channel_attention(x)  # 计算每个通道的权重
            out = x * channel_weight  # 对输入特征进行加权

            # 空间注意力
            max_pool = torch.max(out, dim=1, keepdim=True)[0]  # 最大池化
            avg_pool = torch.mean(out, dim=1, keepdim=True)  # 平均池化
            spatial_weight = self.spatial_attention(torch.cat([max_pool, avg_pool], dim=1))  # 计算每个位置的权重
            x = out * spatial_weight  # 对输入特征进行加权
        return x
