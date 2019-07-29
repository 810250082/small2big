"""
评估正确率
"""
from contain_dataset import ContainDataset, collect_func
import torch.utils.data as tdata
from img_transform import PhotoTransform
from contain_net import ContainNet
import torch
anno_test = 'annot/test.txt'
batch_size = 32
reuse_checkout = 'weight/epoch1.pth'


test_dataset = ContainDataset(anno_test, PhotoTransform())
test_iter = tdata.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collect_func)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

net = ContainNet()
net.load_state_dict(torch.load(reuse_checkout, map_location=lambda storage, loc: storage))

net.to(device=device)

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