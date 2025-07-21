import torch
import torch.nn as nn

"""
门控卷积层是一种特殊的卷积层， 其中输入特征通过两个独立的卷积操作，
一个用于生成主要特征图，另一个用于生成一个门控信号， 该信号控制主要的特征图中的信息流动。
门控机制允许网络在特征表示上引入更多的非线性， 并且通过门控信号动态地控制信号的流动，提高模型的表达能力和适应性
"""

class GatedConv2dWithActivation(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as nn.conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv2dWithActivation, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        for m in self.modules(): #kaiming_normal_: 方法会从标准正态分布（均值为0，标准差为1）中采样权重
            if isinstance(m, nn.Conv2d):#检查当前遍历到的模块m是否是nn.Conv2d类型
                nn.init.kaiming_normal_(m.weight)

    def gated(self, mask):#接受一个掩码（门控信号）
        #return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)


    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)

        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)

        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x