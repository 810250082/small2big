from itertools import product
from math import sqrt
import torch


class PriorBox(object):
    """
    获取锚框
    :param cfg:
    :return:
    """
    def __init__(self, cfg=[]):
        """
        cfg 配置
        :param cfg:
        """
        self.feature_maps = [38, 19, 10, 5, 3, 1]
        self.mult_nums = [8, 16, 32, 64, 100, 300]
        self.origin_size = 300
        self.min_sizes = [30, 60, 111, 162, 213, 264]
        self.max_sizes = [60, 111, 162, 213, 264, 315]
        self.aspects = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.clip = True

    def prior(self):
        """
        产生锚框
        :return:
        """
        box = []
        for k, feature_size in enumerate(self.feature_maps):
            for x, y in product(range(feature_size), repeat=2):
                x, y = x+0.5, y+0.5
                x = x*self.mult_nums[k] / self.origin_size
                y = y*self.mult_nums[k] / self.origin_size
                min_edge = self.min_sizes[k] / self.origin_size
                # 锚框1
                box +=[x, y, min_edge, min_edge]
                # 锚框2
                max_edge = sqrt(min_edge * (self.max_sizes[k] / self.origin_size))
                box += [x, y, max_edge, max_edge]
                # 剩余的锚框
                for aspect in self.aspects[k]:
                    box += [x, y, min_edge*sqrt(aspect), min_edge / sqrt(aspect)]
                    box += [x, y, min_edge / sqrt(aspect), min_edge*sqrt(aspect)]
        output = torch.Tensor(box).view(-1, 4)
        if self.clip:
            output = output.clamp(0, 1)
        return output


if __name__ == '__main__':
    a = PriorBox()
    b = a.prior()
    c = 1