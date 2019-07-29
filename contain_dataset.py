import torch.utils.data as tdata
import torch
import numpy as np
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
        return len(self.annos)

    def __getitem__(self, i):
        target, anchor, points, label = self.parse_anno(self.annos[i])
        # print(label)
        # cv2.imshow('11', target)
        # cv2.waitKey(0)
        # cv2.imshow('111', anchor)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        h, w = anchor.shape[:2]
        x1, y1, x2, y2 = (points * np.array([w, h, w, h])).astype(int)
        anchor = anchor[x1: x2, y1: y2]
        # cv2.imshow('1111', anchor)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        target, anchor, label = self.transform(target, anchor, label)
        target = target[:, :, (2, 1, 0)]
        anchor = anchor[:, :, (2, 1, 0)]

        return torch.tensor(target, dtype=torch.float).permute(2, 1, 0), \
               torch.tensor(anchor, dtype=torch.float).permute(2, 1, 0), \
               label

    def parse_anno(self, anno_txt):
        """
        解析标注
        :param anno:
        :return:
        """
        annos = anno_txt.split(',')
        target = cv2.imread(annos[0])
        anchor = cv2.imread(annos[1])
        points = [float(item) for item in annos[2: 6]]
        label = int(annos[6])
        return target, anchor, points, label


def collect_func(batch):
    photos = []
    labels = []
    for item in batch:
        photos.append(torch.cat([item[0], item[1]], 0).unsqueeze(0))
        labels.append(item[2])
    return torch.cat(photos, 0), torch.tensor(labels, dtype=torch.float).unsqueeze(1)

