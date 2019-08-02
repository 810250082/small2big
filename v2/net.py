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
    def __init__(self):
        """
        初始化
        """
        super(SsdBase, self).__init__()
        self.vgg_out_list = [64, 64, 'M', 128, 128, 'M', 256, 256, 256,
                             'M', 512, 512, 512, 'L2N', 'M', 512, 512,
                             512]
        self.l2_norm = L2Normal(512, 20)

    def conv_base(self, input_dim, out_dim):
        conv = nn.Conv2d(input_dim, out_dim, kernel_size=3, padding=1, stride=1)
        relu = nn.ReLU()
        return conv, relu

    def forward(self, X):
        # vgg base
        input_dim = X.size[1]
        for out_dim in self.vgg_out_list:
            if isinstance(out_dim, int):
                conv, relu = self.conv_base(input_dim, out_dim)
                X = conv(X)
                X = relu(X)
            elif out_dim == 'M':
                maxpooling = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
                X = maxpooling(X)
            elif out_dim == 'L2N':
                X = self.l2_norm(X)

        X = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(X)
        X = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)(X)
        X = nn.ReLU()(X)
        X = nn.Conv2d(1024, 1024, kernel_size=1)(X)
        X = nn.ReLU()(X)

        # muti layer



class L2Normal(nn.Module):
    def __init__(self, channel_num, scale):
        super(L2Normal, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(channel_num))
        nn.init.constant_(self.weight, scale)

    def forward(self, x):
        norm = x.pow(2).sum(1, keepdims=True)
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

if __name__ == '__main__':
    x = torch.rand(10, 3, 112, 150)
    net = TargetExtractNet()

    summary(net, (3, 112, 150), device='cpu')
    # y = net(x)
    # b = 1