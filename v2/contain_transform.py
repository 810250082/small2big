"""
数据集变换
"""
import cv2
import random
import numpy as np


class Resize(object):
    """
    将原图片变为 300*300
    """
    def __init__(self, size=300):
        self.size = size

    def __call__(self, target, origin):
        h, w = origin.shape[:2]
        origin = cv2.resize(origin, (self.size, self.size))
        scale_x, scale_y = self.size/w, self.size/h
        target = cv2.resize(target, dsize=None, fx=scale_x, fy=scale_y)
        return target, origin


class Rotation(object):
    """
    随机翻转
    """
    def __call__(self, target, origin):
        rand_num = random.randint(0, 1)
        if rand_num:
            target = target.transpose((1, 0, 2))
        return target, origin


class Scale(object):
    """
    对目标图片进行缩放
    """
    def __init__(self, rand_min=0.8, rand_max=1.2):
        self.scale = random.uniform(rand_min, rand_max)

    def __call__(self, target, origin):
        target = cv2.resize(target, dsize=None, fx=self.scale, fy=self.scale)
        return target, origin


class TargetScaleTo(object):
    """
    将目标缩放到最短边为 112的大小
    """
    def __init__(self, min_side=112):
        self.min_side = min_side

    def __call__(self, target, origin):
        h, w = target.shape[:2]
        dsize = (self.min_side, int(self.min_side*h/w))
        if h < w:
            dsize = (int(self.min_side* w/h), self.min_side)
        return cv2.resize(target, dsize), origin


class SubtractMeans(object):
    def __init__(self):
        self.mean = np.array([104, 117, 123], dtype=np.float32)

    def __call__(self, target, origin):
        # cv2.imshow('11', target)
        # cv2.waitKey(0)
        # cv2.imshow('12', origin)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        target = target.astype(np.float32)
        origin = origin.astype(np.float32)
        target -= self.mean
        origin -= self.mean
        return target, origin

class DataTrans(object):
    """
    数据集变换类
    """
    def __init__(self):
        self.trans_list = [
            Resize(),
            Rotation(),
            Scale(),
            TargetScaleTo(),
            SubtractMeans()
        ]

    def compose(self, target, origin):
        """
        变换组合
        :param target:
        :param origin:
        :return:
        """
        for tran in self.trans_list:
            target, origin = tran(target, origin)
        return target, origin

    def __call__(self, target, origin):
        return self.compose(target, origin)
