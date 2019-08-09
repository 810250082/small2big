"""
对ssd的改进, 可以定位一张图片在另一张图片中出现的位置
在另一张图像中, 该图片可能已经大小, 已经改变
"""
import torch
from torch import nn
from torchsummary import summary
from v2.prior_box import PriorBox
from v2.config import cfg


class ContainNet(nn.Module):
    def __init__(self):
        super(ContainNet, self).__init__()
        self.base_net = SsdBase()
        self.target_net = TargetExtractNet()
        self.anchar_num = [4, 6, 6, 6, 4, 4]
        # self.cfg = {
        #     # 'num_classes': 2,
        #     # # 'lr_steps': (80000, 100000, 120000),
        #     # # 'max_iter': 120000,
        #     #
        #     'feature_maps': [38, 19, 10, 5, 3, 1],
        #     'min_dim': 300,
        #     'steps': [8, 16, 32, 64, 100, 300],
        #     'min_sizes': [30, 60, 111, 162, 213, 264],
        #     'max_sizes': [60, 111, 162, 213, 264, 315],
        #     'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        #     # 'variance': [0.1, 0.2],
        #     'clip': True,
        #     # 'name': 'VOC',
        # }
        self.cfg = cfg
        self.priorbox = PriorBox(self.cfg)
        self.priors = self.priorbox.prior()

    def forward(self, target, origin):
        """
        前向传播
        :param target:
        :param origin:
        :return:
        """
        target = self.target_net(target)
        muti_features = self.base_net(origin)
        # 将目标特征和原图进行叠加
        superpositions = []
        for feature in muti_features:
            h, w = feature.shape[-2:]
            expand_target = target.unsqueeze(2).unsqueeze(3).expand(-1, -1, h, w)
            superpositions.append(torch.cat((feature, expand_target), dim=1))

        cls_box = []
        offset_box = []
        for k, feature in enumerate(superpositions):
            # 预测是否包含
            conv = nn.Conv2d(feature.shape[1], self.anchar_num[k], kernel_size=3, padding=1, stride=1)
            cls_box.append(conv(feature))
            # 预测偏移量
            conv = nn.Conv2d(feature.shape[1], self.anchar_num[k]*4, kernel_size=3, padding=1, stride=1)
            offset_box.append(conv(feature))

        # 改变形状
        cls_box = torch.cat([item.permute(0, 2, 3, 1).reshape((item.shape[0], -1, 1)) for item in cls_box], dim=1)
        offset_box = torch.cat([item.permute(0, 2, 3, 1).reshape((item.shape[0], -1, 4)) for item in offset_box], dim=1)

        return cls_box, offset_box, self.priors


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

        self.mult_out_list = [
            {'type': 'conv2', 'params': {'out_dim': 256, 'k': 1}},
            {'type': 'relu'},
            {'type': 'conv2', 'params': {'out_dim': 512, 'k': 3, 's': 2, 'p': 1}},
            {'type': 'relu'},
            {'type': 'out'},
            {'type': 'conv2', 'params': {'out_dim': 128, 'k': 1}},
            {'type': 'relu'},
            {'type': 'conv2', 'params': {'out_dim': 256, 'k': 3, 's': 2, 'p': 1}},
            {'type': 'relu'},
            {'type': 'out'},
            {'type': 'conv2', 'params': {'out_dim': 128, 'k': 1}},
            {'type': 'relu'},
            {'type': 'conv2', 'params': {'out_dim': 256, 'k': 3}},
            {'type': 'relu'},
            {'type': 'out'},
            {'type': 'conv2', 'params': {'out_dim': 128, 'k': 1}},
            {'type': 'relu'},
            {'type': 'conv2', 'params': {'out_dim': 256, 'k': 3}},
            {'type': 'relu'},
            {'type': 'out'},
        ]

    def conv_base(self, input_dim, out_dim):
        conv = nn.Conv2d(input_dim, out_dim, kernel_size=3, padding=1, stride=1)
        relu = nn.ReLU()
        return conv, relu

    def forward(self, X):
        out_box = []
        # vgg base
        input_dim = X.shape[1]
        for out_dim in self.vgg_out_list:
            if isinstance(out_dim, int):
                conv, relu = self.conv_base(input_dim, out_dim)
                X = conv(X)
                X = relu(X)
                input_dim = out_dim
            elif out_dim == 'M':
                maxpooling = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
                X = maxpooling(X)
            elif out_dim == 'L2N':
                X1 = self.l2_norm(X)
                out_box.append(X1)

        X = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(X)
        X = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)(X)
        X = nn.ReLU()(X)
        X = nn.Conv2d(1024, 1024, kernel_size=1)(X)
        X = nn.ReLU()(X)
        out_box.append(X)

        # muti layer
        input_dim = X.shape[1]
        for item in self.mult_out_list:
            if item['type'] == 'conv2':
                params = {}
                if 's' in item['params']:
                    params['stride'] = item['params']['s']
                if 'p' in item['params']:
                    params['padding'] = item['params']['p']
                out_dim = item['params']['out_dim']
                conv = nn.Conv2d(input_dim, out_dim,
                                 kernel_size=item['params']['k'],
                                 **params)
                X = conv(X)
                input_dim = out_dim
            elif item['type'] == 'out':
                out_box.append(X)
            elif item['type'] == 'relu':
                relu = nn.ReLU()
                X = relu(X)
        # 返回采集到的层
        return out_box


class L2Normal(nn.Module):
    def __init__(self, channel_num, scale):
        super(L2Normal, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(channel_num))
        nn.init.constant_(self.weight, scale)

    def forward(self, x):
        norm = x.pow(2).sum(1, keepdim=True)
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

if __name__ == '__main__':
    # x = torch.rand(10, 3, 112, 150)
    # net = TargetExtractNet()
    #
    # summary(net, (3, 112, 150), device='cpu')

    net = ContainNet()
    # summary(net, (3, 300, 300), device='cpu')
    x = torch.rand(10, 3, 300, 300)
    target = torch.Tensor(10, 3, 112, 145)
    y = net(target, x)
    # y = net(x)
    b = 1
