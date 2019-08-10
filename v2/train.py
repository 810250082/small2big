"""
训练
"""
from v2.dataset import ContainData
from v2.contain_transform import DataTrans
import torch.utils.data as tdata

# 数据集
def collect_fn(batch):
    pass

train_file = 'annot/train.txt'
dataset = ContainData(train_file, transform=DataTrans())
data_iter = tdata.DataLoader(dataset, shuffle=True, batch_size=16,
                             collate_fn=collect_fn)

# 加载网络

# 训练

# 评估
