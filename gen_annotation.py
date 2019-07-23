"""
generate annotation

the result is a text file

format:
target_img_name,original_img_path,x1,y1,x2,y2,is_contain_target

e.g.

"""
import random
import numpy as np
import os
from collections import defaultdict
from PIL import Image
import shutil


img_path = 'imgs/origin'
target_path = 'imgs/target'
annotation = 'imgs/annot.txt'
# 大小的取值范围
s_list = [0.2, 0.8]
# 纵横比的取值范围
r_list = [0.3, 3]
# 每张图片取目标框的数量
target_num = 6
# 每个目标框对应的正例数量
pos_num = 5
# 每个目标框对应的负例的数量
nav_num = 10


def gen_box(s, r):
    """
    在的图像上随机生成一个 大小为s, 纵横比为r的框
    :param s:
    :param r:
    :return:
    """
    # 随机获取一个中心坐标
    center_x = random.uniform(0.2, 0.8)
    center_y = random.uniform(0.2, 0.8)
    w = np.sqrt(s/r)
    h = np.sqrt(s*r)
    # 左上角和右下角的坐标分别是
    x1, y1 = max(0, center_x - w/2), max(0, center_y - h/2)
    x2, y2 = min(1, center_x + w/2), min(1, center_y + h/2)
    return x1, y1, x2, y2


def iou(box1, box2):
    """
    计算box1 和 box2 的iou
    :param box1:
    :param box2:
    :return:
    """
    inter_area = intersection(box1, box2)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # 并集
    union_area = box2_area + box1_area - inter_area

    return inter_area / union_area


def intersection(box1, box2):
    """
    计算box1, box2 的交集
    :param box1:
    :param box2:
    :return:
    """
    # 交集
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    # 交集
    inter_area = max(inter_y2 - inter_y1, 0) * max(inter_x2 - inter_x1, 0)
    return inter_area


def select_target_imgs(s_list, r_list, n, threshold=0.8):
    """
    从图像上随机选取指定数量的符合要求的目标框
    :param s_list:  可供选择的 s的列表
    :param r_list:  可供选择的r的列表
    :param n:       要选取的目标框数量
    :param threshold:生成的目标框和之前的目标框的阀值不能大于threshold
    :return: n 个目标框
    """
    box = {}
    while len(box) != n:
        s = random.uniform(s_list[0], s_list[1])
        r = random.uniform(r_list[0], r_list[1])
        gen_target = gen_box(s, r)
        # 对于太小的, 要重新生成
        if (gen_target[2] - gen_target[0]) < 0.05 or \
            (gen_target[3] - gen_target[1]) < 0.05:
            continue

        # 对于生成的框, 和已有的目标框太相似, 也要去除
        is_in = False
        for k, target in box.items():
            # 获取交并比
            iou_val = iou(gen_target, target)
            if iou_val > threshold:
                is_in = True
                break
        if is_in:
            continue
        # 存入box
        box[len(box)+1] = gen_target
    return box


def gen_anchor(s_list, r_list):
    """
    获取一个锚框
    :param s_list:
    :param r_list:
    :return:
    """
    s = random.uniform(s_list[0], s_list[1])
    r = random.uniform(r_list[0], r_list[1])
    anchor = gen_box(s, r)
    return anchor


def is_contain_target(anchor, target, threshold=0.5):
    """
    判断锚框是否包含目标框, 如果交集 / 目标框的大小 > threshold, 就认为是包含
    :param anchor:      锚框
    :param target:      目标框
    :param threshold:   阀值, 判断是否包含的依据
    :return:
    """
    # 交集
    inter_area = intersection(anchor, target)
    target_area = (target[2] - target[0]) * (target[3] - target[1])
    if inter_area * 1.0 / target_area > threshold:
        return True
    return False


def gen_anno(img_path, target_path, anno_name, img_num,
             target_s_list, target_r_list, anchor_s_list, anchor_r_list,
             target_num=6, pos_num=5, nav_num=10):
    """
    从img_path中取原始图片上生成target_num个目标框,
    然后对每个目标框生成 pos_num个正例,nav_num个负例
    最后将 目标图片保存到target_path 文件夹中, 将目标框和正例,负例
    都保存到 文件中.
    :param img_path:
    :param target_path:
    :param img_num:         从img_path 获取原始图像的数量
    :param target_num:
    :param pos_num:
    :param nav_num:
    :return:
    """
    # 删除标注和 目标文件夹
    if os.path.exists(anno_name):
        os.remove(anno_name)
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    os.mkdir(target_path)

    # 随机抽取若干张图片
    all_imgs = os.listdir(img_path)
    select_imgs = random.sample(all_imgs, img_num)
    annos = []

    for img_name_with_suffix in select_imgs:
        # 制造目标框和锚框
        # 获取目标框
        targets = select_target_imgs(target_s_list, target_r_list, target_num, threshold=0.5)

        target_anchar_dict = {k: defaultdict(list) for k in targets}
        # # 生成目标框对应的正例和负例
        # while True:
        #     # 是否生成完毕标志位
        #     is_full = True
        #     anchor = gen_anchor((0.2, 0.9), (0.5, 3))
        #     for k, target in targets.items():
        #         if is_contain_target(anchor, target):
        #             # 判断该目标框是否已经存满
        #             if len(target_anchar_dict[k]['pos']) < pos_num:
        #                 is_full = False
        #                 target_anchar_dict[k]['pos'].append(anchor)
        #                 break
        #         else:
        #             if len(target_anchar_dict[k]['nav']) < nav_num:
        #                 is_full = False
        #                 target_anchar_dict[k]['nav'].append(anchor)
        #                 break
        #     if is_full:
        #         break

        for k, target in targets.items():
            while len(target_anchar_dict[k]['pos']) + len(target_anchar_dict[k]['nav']) < pos_num + nav_num:
                anchor = gen_anchor(anchor_s_list, anchor_r_list)
                is_contain = is_contain_target(anchor, target)
                if len(target_anchar_dict[k]['pos']) < pos_num and is_contain:
                    target_anchar_dict[k]['pos'].append(anchor)
                if len(target_anchar_dict[k]['nav']) < nav_num and not is_contain:
                    target_anchar_dict[k]['nav'].append(anchor)

        img_file = os.path.join(img_path, img_name_with_suffix)
        img_name = img_name_with_suffix.split('.')[0]
        img = Image.open(img_file)
        w, h = img.size
        target_name = {}
        # 保存目标框
        for k, target in targets.items():
            target_point = (target * np.array([w, h, w, h])).astype(int)
            # target_img = img[target_point[1]: target_point[3], target_point[0]: target_point[1], :]
            target_img = img.crop(target_point)
            target_name[k] = os.path.join(target_path, "{}_{}.jpg".format(img_name, k))
            # target_img = Image.fromarray(target_img)
            target_img.save(target_name[k])

        # 生成标注
        for k, item in target_anchar_dict.items():
            for label, anchor_list in item.items():
                for anchor in anchor_list:
                    arr = []
                    arr.append(target_name[k])
                    arr.append(img_file)
                    arr.append(','.join(['{:.2f}'.format(point) for point in anchor]))
                    arr.append('1' if label == 'pos' else '0')
                    annos.append(','.join(arr))

    # 将标注写入文件中
    with open(anno_name, 'w') as f:
        f.write('\n'.join(annos))


if __name__ == '__main__':
    gen_anno(img_path, target_path, annotation, 1,
             (0.1, 0.3), (0.5, 2), (0.2, 0.6), (0.5, 2),
             target_num=6, pos_num=5, nav_num=10)
    b = 1
