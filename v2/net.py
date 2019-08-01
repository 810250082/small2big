"""
对ssd的改进, 可以定位一张图片在另一张图片中出现的位置
在另一张图像中, 该图片可能已经大小, 已经改变
"""
import torch
from torch import nn
from torchsummary import summary


class ContainNet(nn.Module):
    def __init__(self):
        super(ContainNet, self).__init__()
        pass

    def forward(self, target, origin):
        """
        前向传播
        :param target:
        :param origin:
        :return:
        """
        pass


class TargetExtractNet(nn.Module):
    """
    目标提取网络
    """
    def __init__(self):
        super(TargetExtractNet, self).__init__()
        output_dims = [64, 128, 256, 256, 512, 512]
        layers = []
        input_dim = 3
        for out_dim in output_dims:
            layers.extend(self.unit(input_dim, out_dim))
            input_dim = out_dim
        self.feature = nn.ModuleList(layers)
        self.output = nn.AdaptiveMaxPool2d(1)

    def unit(self, input_dim, out_dim):
        """
        卷积单元
        :param input_dim:
        :param out_dim:
        :return:
        """
        conv = nn.Conv2d(input_dim, out_dim, kernel_size=3, padding=1, stride=1)
        bn = nn.BatchNorm2d(out_dim)
        relu = nn.ReLU()
        pooling = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        return [conv, bn, relu, pooling]

    def forward(self, target):
        """
        :param target:
        :return:
        """
        for layer in self.feature:
            target = layer(target)
        target = self.output(target)
        C = target.size()[1]
        return target.view(-1, C)


class SsdBase(nn.Module):
    """
    ssd 基础网络
    """
    pass


if __name__ == '__main__':
    x = torch.rand(10, 3, 112, 150)
    net = TargetExtractNet()

    summary(net, (3, 112, 150), device='cpu')
    # y = net(x)
    # b = 1