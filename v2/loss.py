"""
损失计算
"""
import torch
from torch import nn


# def allot_prior(prior, target, threshold=0.5):
#     """
#     计算锚框和目标框
#     :param prior:       (anchar_num, 4)
#     :param target:      (batch, 4)
#     :return:
#             cls:        (batch, anchar_num, 1)
#             mask:       (batch, anchar_num*4)
#             offset:     (batch, anchar_num*4)
#     """
#     # 将 锚框转为都是坐标的形式
#     prior_point = trans_points(prior)
#     # 将锚框和目标框都转为 批量,锚框, 4 的形状
#
#     inter_x1 = torch.max(target[0], prior_point[:, 0])
#     inter_y1 = torch.max(target[1], prior_point[:, 1])
#     inter_x2 = torch.min(target[2], prior_point[:, 2])
#     inter_y2 = torch.min(target[3], prior_point[:, 3])
#     inter_w, inter_h = torch.min(torch.Tensor([0]), inter_x2-inter_x1), torch.min(torch.Tensor([0]), inter_y2-inter_y1)
#
#     inter_area = inter_h * inter_w
#
#     union_area = (target[2] - target[0]) * (target[3] - target[1]) + (prior_point[:, 2] - prior_point[:, 0]) * (prior_point[:, 3] - prior_point[1])
#
#     iou = inter_area / (union_area - inter_area)
#     cls = iou.gt(threshold)
#     # 获取 mask
#     mask = cls.expand(-1, 4).view(-1)
#     # 获取偏移量
#     offset = get_offset(target, prior).view(-1)
#
#     return cls, mask, mask * offset


def allot_prior(prior, target, threshold=0.5):
    """
    计算锚框和目标框
    :param prior:       (anchar_num, 4)
    :param target:      (batch, 4)
    :return:
            cls:        (batch, anchar_num)
            mask:       (batch, anchar_num*4)
            offset:     (batch, anchar_num*4)
    """
    # 将 锚框转为都是坐标的形式
    prior_point = trans_points(prior)
    # 将锚框和目标框都转为 批量,锚框, 4 的形状
    prior_point = prior_point.unsqueeze(0).expand(target.shape[0], -1, -1)
    target = target.unsqueeze(1).expand_as(prior_point)

    inter_x1 = torch.max(target[:, :, 0], prior_point[:, :, 0])
    inter_y1 = torch.max(target[:, :, 1], prior_point[:, :, 1])
    inter_x2 = torch.min(target[:, :, 2], prior_point[:, :, 2])
    inter_y2 = torch.min(target[:, :, 3], prior_point[:, :, 3])
    inter_w, inter_h = torch.max(torch.Tensor([0]), inter_x2-inter_x1), torch.max(torch.Tensor([0]), inter_y2-inter_y1)

    # shape: batch, anchar
    inter_area = inter_h * inter_w

    union_area = (target[:, :, 2] - target[:, :, 0]) * (target[:, :, 3] - target[:, :, 1]) + (prior_point[:, :, 2] - prior_point[:, :, 0]) * (prior_point[:, :, 3] - prior_point[:, :, 1])

    iou = inter_area / (union_area - inter_area)
    cls = iou.gt(threshold)
    # 获取 mask
    mask = cls.unsqueeze(-1).expand(-1, -1, 4).reshape(cls.shape[0], -1)
    # 获取偏移量
    offset = get_offset(target, prior).reshape(cls.shape[0], -1)

    return cls, mask, mask * offset


def trans_points(prior):
    """
    将锚框转为坐标形式
    :param prior:
    :return:
    """
    return torch.stack((prior[:, :2] - prior[:, 2:]/2, prior[:, :2] + prior[:, :2]/2), dim=1)


def get_offset(target, prior):
    """
    获取偏移量
    :param target:
    :param prior:
    :return:
    """
    return torch.stack(((target[:, :, :2] - prior[:, :, :2]) / prior[:, :, 2:], torch.log(target[:, :, 2:]/prior[:, :, 2:])), dim=2)


def calculate_loss(pre_offset, pre_cls, prior, target):
    """
    计算损失
    :param pre_offset:  预测偏移量       (batch, anchar_num, 1)
    :param pre_cls:     预测类别        (batch, anchar_num, 4)
    :param prior:       锚框坐标            (anchar_num, 4)
    :param target:      真是目标坐标      (batch, 4)
    :return:
    """
    cls, mask, offset = allot_prior(prior, target)

    # 类别损失
    cls_loss_fun = nn.BCEWithLogitsLoss()
    cls_loss = cls_loss_fun(pre_cls, cls)

    # 偏移损失
    offset_loss_func = nn.SmoothL1Loss()
    offset_loss = offset_loss_func(pre_offset * mask, offset)
    l = cls_loss + offset_loss
    return l