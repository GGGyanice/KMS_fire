# -*- coding:utf-8 -*-
"""
Time : 2020/10/15 11:13
Author : Kexin Guan
Decs ：
"""

import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt


def pic_hist(img, name_str):
    # plt.hist(img.ravel(), 256, [0, 256])
    # plt.savefig('./pic/dc_hist/' + str(name_str) + '.png')
    # plt.clf()
    # plt.show()

    color = ('blue', 'green', 'red')
    for i, color in enumerate(color):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.savefig('./pic/cv_hist/' + str(name_str) + '.png')
    # plt.clf()
    # plt.show()
    

def dicrease_color(img):
    return img // 64 * 64 + 32


def flame_rect(image, name_str):
    # 阈值分割
    ret, thresh = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 120, 255, cv2.THRESH_TOZERO)
    thresh[thresh > 120] = 255
    # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 开运算, 先腐蚀后膨胀
    open_ = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

    # hierarchy:[后、前、子、父轮廓的索引]
    contours, hierarchy = cv2.findContours(open_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = np.zeros((1, len(contours)))
    # 取面积前5大的轮廓
    for m in range(len(contours)):
        areas[0, m] = cv2.contourArea(contours[m])
        for j in range(contours[m].shape[0]):
            for n in range(contours[m][j].shape[0]):
                cv2.circle(image, (contours[m][j][n][0], contours[m][j][n][1]), 1, (0, 255, 0), 2)
    areas = np.argsort(-areas)[0][:5]

    num = 6  # 火焰面积、矩形xy、长宽、角度
    features = np.zeros((1, 5*num+1))
    for i in range(areas.shape[0]):
        # (x, y), (长，宽)，(旋转角度)
        min_rect = cv2.minAreaRect(contours[areas[i]])
        line_area = cv2.contourArea(contours[areas[i]])
        coordinate, length, angle = np.array(min_rect[0]), np.array(min_rect[1]), np.array(min_rect[2])
        # 0右下→1左下→2左上→3右上
        rect_points = cv2.boxPoints(min_rect)
        features[0, 0] = str(name_str)
        features[0, num*i+1:num*(i+1)+1] = np.hstack([line_area, coordinate, length, angle]).reshape(1, -1)
        print(features[0, num*i+1:num*(i+1)+1])
        # 画矩形
        rect_points = np.int0(rect_points)
        image = cv2.drawContours(image, [rect_points], 0, (255, 255, 255), 2)
        
    return features, image


if __name__ == '__main__':
    rect_csv = './doc/rect.csv'
    dir = "./pic/target"
    img_list = os.listdir(dir)
    img_list.sort(key=lambda x: int(x.replace("target.mp4_", "").split('.')[0]))

    for n in range(len(img_list)):
        img_path = os.path.join(dir, img_list[n])
        image = cv2.imread(img_path)
        name_str = img_path[-14:-8]
        print(name_str)
        
        rect_feature, img = flame_rect(image, name_str)
        pd.DataFrame(rect_feature).to_csv(rect_csv, mode='a', index=False, header=False)
        cv2.imwrite("./pic/rect/" + str(name_str) + ".jpg", img)
        # cv2.imshow(name_str, img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    col = ["name", "rect0_area", "rect0_x", "rect0_y", "rect0_height", "rect0_width", "rect0_angle",
           "rect1_area,", "rect1_x", "rect1_y", "rect1_height", "rect1_width", "rect1_angle",
           "rect2_area,", "rect2_x", "rect2_y", "rect2_height", "rect2_width", "rect2_angle",
           "rect3_area,", "rect3_x", "rect3_y", "rect3_height", "rect3_width", "rect3_angle",
           "rect4_area,", "rect4_x", "rect4_y", "rect4_height", "rect4_width", "rect4_angle"]
    data = pd.DataFrame(pd.read_csv(rect_csv, header=None).values, columns=col).to_csv(rect_csv, index=False)
