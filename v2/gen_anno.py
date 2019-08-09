"""
制作目标数据集
"""
import os
import cv2
import random
import numpy as np
import shutil
import math


def gen_box(s, r):
    """
    在的图像上随机生成一个 大小为s, 纵横比为r的框
    :param s:
    :param r:
    :return:
    """
    # 随机获取一个中心坐标
    center_x = round(random.uniform(0.2, 0.8), 2)
    center_y = round(random.uniform(0.2, 0.8), 2)
    w = np.sqrt(s/r)
    h = np.sqrt(s*r)
    # 左上角和右下角的坐标分别是
    x1, y1 = max(0, center_x - w/2), max(0, center_y - h/2)
    x2, y2 = min(1, center_x + w/2), min(1, center_y + h/2)
    return x1, y1, x2, y2


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


def gen_anno(img_base_path, origin_num, anno_base_path='../annot', target_s=(0.1, 0.3), target_r=(0.5, 2), per_target_num=6, split_rate=(0.8, 0.8)):
    """
    生成标注文本, 并且分割成训练集,验证集, 测试集
    :param img_base_path:   图片文件夹
    :param origin_num:      获取原始图片数量
    :param target_s:        生成的目标框大小
    :param target_r:        生成的目标框纵横比
    :param per_target_num:  每张原始图片生成目标框数量
    :return:
    """
    # 选取指定数量的原始图片
    origin_path = os.path.join(img_base_path, 'origin')
    target_path = os.path.join(img_base_path, 'target')
    train_anno_name = os.path.join(anno_base_path, 'train.txt')
    val_anno_name = os.path.join(anno_base_path, 'val.txt')
    test_anno_name = os.path.join(anno_base_path, 'test.txt')
    # 删除目标图像
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    os.mkdir(target_path)
    # 删除标注文件夹
    if os.path.exists(anno_base_path):
        shutil.rmtree(anno_base_path)
    os.mkdir(anno_base_path)

    all_origins = os.listdir(origin_path)
    select_origins = all_origins[:origin_num]

    data = []
    for img_suffix in select_origins:
        img_name = img_suffix.split('.')[0].strip()
        img_path = os.path.join(origin_path, img_suffix)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        # 随机选择指定个目标框
        target_points = select_target_imgs(target_s, target_r, per_target_num)


        for ind, point in target_points.items():
            anno = []
            x1, y1, x2, y2 = (point * np.array([w, h, w, h])).astype(int)
            target = img[y1: y2, x1: x2]
            # cv2.imshow('10', img)
            # cv2.waitKey(0)
            # cv2.imshow('11', target)
            # cv2.waitKey(0)
            # random_trans = random.uniform(0.7, 1.3)
            # print(random_trans)
            # target = cv2.resize(target, None, fx=random_trans, fy=random_trans)
            # cv2.imshow('12', target)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            target_name = os.path.join(target_path, '{}_{}.jpg'.format(img_name, ind))
            cv2.imwrite(target_name, target)
            anno.append(target_name)
            anno.append(img_path)
            anno.append(",".join(['{:.2f}'.format(item) for item in point]))
            # 保存一个样本
            data.append(','.join(anno))

    total_num = len(data)
    train_val_ind = math.floor(split_rate[0] * total_num)
    train_ind = math.floor(split_rate[1] * train_val_ind)
    train_data = data[:train_ind]
    val_data = data[train_ind: train_val_ind]
    test_data = data[train_val_ind:]

    # 将所有的样本保存到文件中
    with open(train_anno_name, 'w') as f:
        f.write('\n'.join(train_data))
    with open(val_anno_name, 'w') as f:
        f.write('\n'.join(val_data))
    with open(test_anno_name, 'w') as f:
        f.write('\n'.join(test_data))


if __name__ == '__main__':
    gen_anno('../imgs', 20)
    print('over')
