"""
判断是否包含小图片深度网络
"""


import torch.utils.data as tdata
import torch
# from PIL import Image
import cv2
# 制作数据集


class ContainDataset(tdata.Dataset):
    """
    数据集类
    """
    def __init__(self, anno_name, transform):
        with open(anno_name, 'r') as f:
            self.annos = [item.strip() for item in f.readlines()]
        self.transform = transform

    def __len__(self):
        len(self.annos)

    def __getitem__(self, i):
        target, anchor, _, label = self.parse_anno(self.annos[i])
        target, anchor, label = self.transform(target, anchor, label)
        return target, anchor, label

    def parse_anno(self, anno_txt):
        """
        解析标注
        :param anno:
        :return:
        """
        annos = ','.split(anno_txt)
        target = cv2.imread(annos[0])
        anchor = cv2.imread(annos[1])
        points = annos[2: 6]
        label = annos[6]
        return target, anchor, points, label

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

