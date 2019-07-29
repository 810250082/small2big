"""
训练
"""

import torch
from torch import nn, optim
from contain_net import ContainNet
import torch.utils.data as tdata
from img_transform import PhotoTransform
from contain_dataset import ContainDataset, collect_func


anno_train = 'annot/train.txt'
anno_test = 'annot/test.txt'
epochs = 3
batch_size = 32
lr = 0.0001
reuse_checkout = 'weight/epoch2_1.pth'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

net = ContainNet()
# 加载模型参数
if reuse_checkout:
    net.load_state_dict(torch.load(reuse_checkout, map_location=lambda storage, loc: storage))

net.to(device=device)
# summary(net, (6, 300, 300))
# 损失函数
loss = nn.BCELoss()
trainer = optim.Adam(net.parameters(), lr=lr)


train_dataset = ContainDataset(anno_train, PhotoTransform())
train_iter = tdata.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collect_func)

test_dataset = ContainDataset(anno_test, PhotoTransform())
test_iter = tdata.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collect_func)


def train(data_iter, net, loss, trainer, epochs):
    for epoch in range(epochs):
        for i, (X, y) in enumerate(data_iter):
            X = X.to(device)
            y = y.to(device)
            net.zero_grad()
            y_pre = net(X)
            l = loss(y_pre, y)
            l.backward()
            trainer.step()
            if i % 5 == 0:
                print('epoch {}/{}, loss {}'.format(epoch, i, l))
        # 保存模型参数
        torch.save(net.state_dict(), 'weight/epoch{}.pth'.format(epoch))


train(train_iter, net, loss, trainer, epochs)


# 评估
net.eval()
total = 0.0
correct_num = 0.0
for X, y in test_iter:
    X = X.to(device)
    y = y.to(device)
    with torch.no_grad():
        y_hat = net(X)
    y_hat = y_hat >= 0.5
    y_hat = y_hat.type_as(y)
    correct_num += (y==y_hat).sum(dtype=torch.float)
    total += len(y_hat)

print('correct rate {}'.format(correct_num / total))
b = 1
