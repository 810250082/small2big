import numpy as np
import cv2
import matplotlib.pyplot as plt


def manhattan_distance(a, b):
    """
    计算曼哈顿记录
    :param a:
    :param b:
    :return:
    """
    return np.sum(np.abs(a - b))


big_img_path = 'one.jpg'
small_img_path = 'small_one.jpg'
coord = (0.3, 0.3, 0.5, 0.55)
# coord = (0, 0, 0.2, 0.3)

big_img = cv2.imread(big_img_path, cv2.IMREAD_COLOR)
big_img = cv2.cvtColor(big_img, cv2.COLOR_BGR2RGB)
plt.imshow(big_img)
plt.show()

h, w, c = big_img.shape
y1 = int(coord[1]*h)
y2 = int(coord[3]*h)
x1 = int(coord[0]*w)
x2 = int(coord[2]*w)
print(x1, y1, x2, y2)
small_img = big_img[y1: y2, x1: x2, :]
plt.imshow(small_img)
plt.show()

small_img = cv2.cvtColor(small_img, cv2.COLOR_RGB2BGR)
cv2.imwrite(small_img_path, small_img)


big_img = cv2.imread(big_img_path, cv2.IMREAD_COLOR)
big_img = cv2.cvtColor(big_img, cv2.COLOR_BGR2RGB).astype(np.float)
small_img = cv2.imread(small_img_path, cv2.IMREAD_COLOR)
small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB).astype(np.float)

b_h, b_w, b_c = big_img.shape
s_h, s_w, s_c = small_img.shape

# box = []
# for i in range(b_w - s_w + 1):
#     for j in range(b_h - s_h + 1):
#         pre_img = big_img[j: j+s_h, i: i+s_w, :]
#         norm = np.sum(np.square(pre_img - small_img))
#         if norm < 0.01:
#             box.append((i, j))
#
#
# plt.imshow(big_img)
# curr_axis = plt.gca()
# for item in box:
#     rect = plt.Rectangle((item[0], item[1]), s_w, s_h, fill=False, linewidth=2)
#     curr_axis.add_patch(rect)
#
# plt.show()

X = []
Y = []
k = 0
for i in range(b_w - s_w + 1):
    print('new row i: {}'.format(i))
    if i != 67:
        continue
    for j in range(b_h - s_h + 1):
        print(k)
        X.append(k)
        pre_img = big_img[j: j+s_h, i: i+s_w, :]
        Y.append(manhattan_distance(pre_img, small_img))
        k += 1

plt.plot(X, Y)

plt.show()