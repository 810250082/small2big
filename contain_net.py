"""
判断是否包含小图片深度网络
"""


import torch.utils.data as tdata
import torch
# from PIL import Image
import cv2
from img_transform import PhotoTransform
from torch import nn, optim
from torchsummary import summary
import numpy as np


class ContainNet(nn.Module):
    def __init__(self):
        """
        初始化层
        """
        super(ContainNet, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpooling1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False)
        # self.conv2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.maxpooling2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False)
        # self.conv3_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.maxpooling3 = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv4 = nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.maxpooling4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.maxpooling5 = nn.MaxPool2d(2, 2)

        self.conv6 = nn.Conv2d(512, 512, 3, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.maxpooling6 = nn.MaxPool2d(3, 2, 1)

        self.conv7 = nn.Conv2d(512, 1, 3)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        for i in range(1, 7):
            conv = getattr(self, 'conv{}'.format(i))
            bn = getattr(self, 'bn{}'.format(i))
            maxpooling = getattr(self, 'maxpooling{}'.format(i))
            # x = maxpooling(nn.ReLU(bn(conv(x))))
            x = maxpooling(nn.functional.relu(bn(conv(x))))
            # x = self.maxpooling1(nn.ReLU(self.bn1(self.conv1(x))))
        x = torch.sigmoid(self.conv7(x))
        return x.view(-1, 1)
#
#
# anno_train = 'annot/train.txt'
# anno_test = 'annot/test.txt'
# epochs = 5
# batch_size = 32
# lr = 0.0001
# reuse_checkout = 'weight/epoch4.pth'
#
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#
# net = ContainNet()
# # 加载模型参数
# if reuse_checkout:
#     net.load_state_dict(torch.load(reuse_checkout, map_location=lambda storage, loc: storage))
#
# net.to(device=device)
# # summary(net, (6, 300, 300))
# # 损失函数
# loss = nn.BCELoss()
# trainer = optim.Adam(net.parameters(), lr=lr)
#
# train_dataset = ContainDataset(anno_train, PhotoTransform())
# train_iter = tdata.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collect_func)
#
# test_dataset = ContainDataset(anno_test, PhotoTransform())
# test_iter = tdata.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collect_func)
#
#
# def train(data_iter, net, loss, trainer, epochs):
#     for epoch in range(epochs):
#         for i, (X, y) in enumerate(data_iter):
#             X = X.to(device)
#             y = y.to(device)
#             net.zero_grad()
#             y_pre = net(X)
#             l = loss(y_pre, y)
#             l.backward()
#             trainer.step()
#             if i % 5 == 0:
#                 print('epoch {}/{}, loss {}'.format(epoch, i, l))
#         # 保存模型参数
#         torch.save(net.state_dict(), 'weight/epoch{}.pth'.format(epoch))
#
#
# train(train_iter, net, loss, trainer, epochs)


if __name__ == '__main__':
    # import numpy as np
    # a = torch.tensor(1)
    # b = torch.from_numpy(np.array([[1, 2], [3, 4]]))
    # dataset = ContainDataset('imgs/annot.txt', PhotoTransform())
    # data_iter = tdata.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collect_func)
    # for X, y in data_iter:
    #     X
    # c = 1

    # x = torch.rand(1, 6, 300, 300)
    # net = ContainNet()
    # y = net(x)

    # train()



#
# from torch.utils.data import TensorDataset
# from torch.utils.data import DataLoader
#
#
# class SimpleCustomBatch:
#     def __init__(self, data):
#         transposed_data = list(zip(*data))
#         self.inp = torch.stack(transposed_data[0], 0)
#         self.tgt = torch.stack(transposed_data[1], 0)
#
#     def pin_memory(self):
#         self.inp = self.inp.pin_memory()
#         self.tgt = self.tgt.pin_memory()
#         return self
#
# def collate_wrapper(batch):
#     return SimpleCustomBatch(batch)
#
# inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
# tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
# dataset = TensorDataset(inps, tgts)
#
# loader = DataLoader(dataset, batch_size=2, collate_fn=collate_wrapper,
#                     pin_memory=True)
#
# for batch_ndx, sample in enumerate(loader):
#     print(sample.inp.is_pinned())
#     print(sample.tgt.is_pinned())
    pass
