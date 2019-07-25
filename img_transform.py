"""
图片变换
"""
# from torchvision import transforms
import cv2
import random
import numpy as np


class Compose(object):
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, target, anchor, label):
        for t in self.transforms:
            target, anchor, label = t(target, anchor, label)
        return target, anchor, label


class ExpandTarget(object):
    """
    放大目标框
    """
    def __init__(self, low=0.8, up=1.1):
        self.scale = random.uniform(low, up)

    def __call__(self, target, anchor, label):
        # 等比例放大 或 缩小
        target = cv2.resize(target, None, fx=self.scale, fy=self.scale)
        return target, anchor, label


class ExpandTwoPic(object):
    """
    放大两张图片
    """
    def __init__(self, means):
        self.means = means

    def __call__(self, target, anchor, label):
        max_h = max(target.shape[0], anchor.shape[0])
        max_w = max(target.shape[1], anchor.shape[1])
        target_mask = np.ones((max_h, max_w, 3)) * self.means
        anchor_mask = target_mask.copy()
        target_mask[0: target.shape[0], 0: target.shape[1]] = target
        anchor_mask[0: anchor.shape[0], 0: anchor.shape[1]] = anchor
        return target_mask.astype(np.float32), anchor_mask.astype(np.float32), label


class SubtractMean(object):
    """
    减去平均值
    """
    def __init__(self, means):
        self.means = means

    def __call__(self, target, anchor, label):
        target = target - self.means
        anchor = anchor - self.means
        return target, anchor, label


class PhotoTransform(object):
    def __init__(self, means=(104, 117, 123)):
        self.means = means
        self.compose = Compose([
            ExpandTarget(),
            ExpandTwoPic(self.means),
            SubtractMean(self.means)
        ])

    def __call__(self, target, anchor, label):
        """
        变换
        :param target:
        :param anchor:
        :param label:
        :return:
        """
        return self.compose(target, anchor, label)


if __name__ == '__main__':
    import cv2
    anchor = cv2.imread('imgs/origin/one.jpg')
    target = cv2.imread('imgs/target/one_1.jpg')

    trans = PhotoTransform(means=(104, 117, 123))
    a, b, c = trans(target, anchor, 1)
    d = 1