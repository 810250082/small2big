"""
训练
"""
from v2.dataset import ContainData
from v2.contain_transform import DataTrans
import torch.utils.data as tdata
import torch
import cv2
from v2.net import ContainNet
from v2.loss import calculate_loss
from torch.optim import Adam


# 数据集
def collect_fn(batch):
    target = torch.Tensor(batch[0][0]).unsqueeze(0).permute(0, 3, 1, 2)
    origin = torch.Tensor(batch[0][1]).unsqueeze(0).permute(0, 3, 1, 2)
    point = torch.Tensor(batch[0][2])
    return target, origin, point

train_file = '../annot/train.txt'
dataset = ContainData(train_file, transform=DataTrans())
data_iter = tdata.DataLoader(dataset, shuffle=True, batch_size=1,
                             collate_fn=collect_fn)


# 加载网络
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
net = ContainNet()
net.to(device)
trainer = Adam(net.parameters(), lr=0.001)

# 训练
for target, origin, point in data_iter:
    target = target.to(device)
    origin = origin.to(device)
    point = point.to(device)
    pre_cls, pre_offset, priors = net(target, origin)
    l = calculate_loss(pre_offset, pre_cls, priors, point)
    l.backward()
    trainer.step()
# 评估
