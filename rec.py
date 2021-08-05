# -*- coding: utf-8 -*-

"""
@author:
@time:
@description:
参考：
https://zhuanlan.zhihu.com/p/115317146
https://www.cnblogs.com/gezhuangzhuang/p/10724769.html
https://cloud.tencent.com/developer/article/1544624?from=14588
"""

import cv2
from scipy import signal
from matplotlib import pyplot as plt


def pic2gray(pic_path, new_path=''):
    # 转灰度图
    pic_path_rgb = cv2.imread(pic_path)
    pic_path_gray = cv2.cvtColor(pic_path_rgb, cv2.COLOR_BGR2GRAY)
    if new_path:
        cv2.imwrite(new_path, pic_path_gray)
    return pic_path_gray


def canny_edge1(image_array, new_path=''):
    # 灰度图锐化
    # can = cv2.Canny(image_array, threshold1=200, threshold2=300)
    can = cv2.Canny(image_array, threshold1=100, threshold2=100)
    if new_path:
        cv2.imwrite(new_path, can)
    return can


def canny_edge2(image_array, new_path=''):
    # 灰度图锐化
    can = cv2.Canny(image_array, threshold1=300, threshold2=300)
    if new_path:
        cv2.imwrite(new_path, can)
    return can


def clear_white(pic_path, new_path=''):
    # 清除图片空白区域，主要清除滑块的空白
    img_obj = cv2.imread(pic_path)
    rows, cols, channel = img_obj.shape
    min_x, min_y, max_x, max_y = 255, 255, 0, 0
    for x in range(1, rows):
        for y in range(1, cols):
            t = set(img_obj[x, y])
            if len(t) >= 2:
                if x <= min_x:
                    min_x = x
                elif x >= max_x:
                    max_x = x
                if y <= min_y:
                    min_y = y
                elif y >= max_y:
                    max_y = y
    img1_obj = img_obj[min_x: max_x, min_y: max_y]
    if new_path:
        cv2.imwrite(new_path, img1_obj)
    return img1_obj


def convolve2d(bg_array, filter_array):
    # 计算2D卷积 参考：https://blog.csdn.net/m0_38007695/article/details/82794454
    bg_h, bg_w = bg_array.shape[:2]
    filter_h, filter_w = filter_array.shape[:2]
    c_full = signal.convolve2d(bg_array, filter_array, mode='full')
    kr, kc = filter_h // 2, filter_w // 2
    c_same = c_full[
             filter_h - kr - 1: bg_h + filter_h - kr - 1,
             filter_w - kc - 1: bg_w + filter_w - kc - 1,
    ]
    return c_same


def find_max_point(arrays, search_on_horizontal_center=False):
    # 找二维数组中最大的点
    max_point = 0
    max_point_pos = None
    array_rows, array_cols = arrays.shape
    if search_on_horizontal_center:
        for col in range(array_cols):
            if arrays[array_rows // 2, col] > max_point:
                max_point = arrays[array_rows // 2, col]
                max_point_pos = col, array_rows // 2
    else:
        for row in range(array_rows):
            for col in range(array_cols):
                if arrays[row, col] > max_point:
                    max_point = arrays[row, col]
                    max_point_pos = col, row
    return max_point_pos


def main():
    # 带有缺口的图片
    path1 = r"C:\Users\admin\Desktop\fsdownload\jietu_1.png"
    # 模板图片
    path2 = r"C:\Users\admin\Desktop\fsdownload\jietu_2.png"

    # 获取裁剪后的缺口图片
    clear_path = r"C:\Users\admin\Desktop\fsdownload\jietu_2_clear.png"
    clear_white(path2, clear_path)

    # 获取灰度图
    gray_path = r"C:\Users\admin\Desktop\fsdownload\jietu_1_gray.png"
    image_gray1 = pic2gray(path1, gray_path)
    gray_path = r"C:\Users\admin\Desktop\fsdownload\jietu_2_gray.png"
    image_gray2 = pic2gray(clear_path, gray_path)

    # 获取锐化后的图片，背景图片和缺口图片的锐化度不同
    can_path1 = r"C:\Users\admin\Desktop\fsdownload\jietu_1_can.png"
    bg = canny_edge1(image_gray1, can_path1)
    can_path2 = r"C:\Users\admin\Desktop\fsdownload\jietu_2_can.png"
    fil = canny_edge2(image_gray2, can_path2)

    # # 卷积匹配
    # c_same = convolve2d(bg, fil)
    # max_point_pos = find_max_point(c_same)
    # print(max_point_pos)

    img = cv2.imread(can_path1)
    template = cv2.imread(can_path2)
    # 实行缺口匹配算法
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(min_val, max_val, min_loc, max_loc)

    h, w = template.shape[:2]
    # h1, w1 = img.shape[:2]
    # print(h1, w1)

    left_top = max_loc  # 左上角
    right_bottom = (left_top[0] + w, left_top[1] + h)  # 右下角
    cv2.rectangle(img, left_top, right_bottom, 255, 2)  # 画出矩形位置

    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == "__main__":
    main()
