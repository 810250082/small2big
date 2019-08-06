"""
损失计算
"""
import torch


def iou_matrix(prior, target):
    """
    计算锚框和目标框的iou矩阵
    :param prior:
    :param target:
    :return:
    """
    inter_x1 = torch.max(target[:, 0], prior[:, 0])
    inter_y1 = torch.max(target[:, 1], prior[:, 1])
    inter_x2 = torch.min(target[:, 2], prior[:, 2])
    inter_y2 = torch.min(target[:, 3], prior[:, 3])
    inter_w, inter_h = torch.min(torch.Tensor([0]), inter_x2-inter_x1), torch.min(torch.Tensor([0]), inter_y2-inter_y1)

    inter_area = inter_h * inter_w

    union_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    iou = inter_area / (union_area - inter_area)
    return iou


def calculate_loss(pre_offset, pre_cls, prior, target):
    """
    计算损失
    :param pre_offset:  预测偏移量
    :param pre_cls:     预测类别
    :param prior:       锚框坐标
    :param target:      真是目标坐标
    :return:
    """
    # 将锚框转为坐标形式
    # point_prior =
    # #为每一个锚框分配类别和偏移量
    # iou_matrix(prior, )
    # 类别损失
    # 偏移量损失
    pass
