# coding: utf-8
"""
Time: 2020/10/13 14:15
Author: Kexin Guan
Desc: According to the paper
"""
import copy
from multiprocessing import Pool
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
# from PIL import Image


# img_path = "./pic/20200904_005627/20200904_005627_10s/20200904_005627.mp4_000000.006.jpg"
# img = Image.open(img_path)
# # img.show()
# gray = img.convert('L')  # 转换成灰度
# # gray.show()


def gray_use_formula(image_path):
    """
    用公式转成灰度图像
    :param image_path:
    :return: type: ndarray
             shape: (576, 960)
    """
    image = cv2.cvtColor(cv2.imread(image_path).astype(np.uint8), cv2.COLOR_BGR2RGB)  # BGR to RGB
    gray_image = 0.11*image[:, :, 0] + 0.59*image[:, :, 1] + 0.30*image[:, :, 2]
    return gray_image


def step_function(x, value):
    return np.array(x-value > 0, dtype=np.int)


def flame_area(gray_image, value):
    """

    :param gray_image:灰度化后的图
    :param value:阈值 int
    :return:
    """
    z = step_function(gray_image, value)
    return np.sum(np.sum(z, axis=0))/(z.shape[0]*z.shape[1])


def flame_coordinate(g_step, gray_image):
    """
    求火焰质心坐标
    :param g_step: 0-1矩阵
    :param gray_image:灰度图像
    :return: (float, float)
    """
    z = g_step * gray_image
    z_x, z_y = copy.deepcopy(z), copy.deepcopy(z)

    for m in range(z_x.shape[0]):
        z_x[m] = z_x[m] * m

    z_y = z_y.T
    for j in range(z_y.shape[0]):
        z_y[j] = z_y[j] * j
    z_y = z_y.T

    coordinate_x = np.sum(np.sum(z_x, axis=0)) / (np.sum(np.sum(z, axis=0)))
    coordinate_y = np.sum(np.sum(z_y, axis=0)) / (np.sum(np.sum(z, axis=0)))
    return coordinate_y, coordinate_x


def get_variance(step_num, row):
    """
    单位时间属性列方差计算
    :param step_num: each min在10s矩阵中的index， array
    :param row: 图像属性row
    :return: array
    """
    flame_v = []
    for i in range(0, step_num.shape[0] - 1):
        flame_v.append(np.var(row[step_num[i] + 1:step_num[i + 1] + 1], ddof=1))
    return np.array(flame_v).reshape(-1, 1)


def flame_front(img_path, name_str):
    image = cv2.imread(img_path)

    # 阈值分割
    ret, thresh = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 120, 255, cv2.THRESH_TOZERO)
    thresh[thresh > 120] = 255
    # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 开运算, 先腐蚀后膨胀
    open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, hierarchy = cv2.findContours(open, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    q = []
    for m in range(len(contours)):
        for j in range(contours[m].shape[0]):
            for n in range(contours[m][j].shape[0]):
                q.append(contours[m][j][n][1])
                # cv2.circle(image, (contours[m][j][i][0], contours[m][j][i][1]), 2, (0, 255, 0), 3)

    cv2.drawContours(image, contours, -1, (225, 225, 0), 2)
    cv2.line(image, (0, int(np.mean(q))), (image.shape[1], int(np.mean(q))), (0, 0, 255), 2, 4)
    cv2.imwrite("./doc/target/flame_front/" + str(name_str) + ".jpg", image)

    return np.hstack([str(name_str), np.mean(q)]).reshape(1, -1)


def main_function(path, item, flame_value, high_tem, second_csv):

    img_path = os.path.join(path, item)
    name_str = img_path[-14:-8]

    front_data = flame_front(img_path, name_str)

    gray_img = gray_use_formula(img_path)  # 灰度化
    plt.imshow(gray_img, cmap='gray')
    plt.savefig('./pic/gray/' + str(name_str) + '.png')
    g_mean = np.mean(gray_img)  # 灰度均值

    effective_flame = flame_area(gray_img, flame_value)  # 火焰面积

    high_tem_flame = flame_area(gray_img, high_tem)  # 高温率

    g = step_function(gray_img, 80)
    y, x = flame_coordinate(g, gray_img)  # 火焰质心

    img = cv2.imread(img_path)

    result = np.argwhere(g == 1)
    for j in range(result.shape[0]):
        cv2.circle(img, (result[j, 1], result[j, 0]), 2, (255, 255, 255), 3)
    cv2.circle(img, (int(y), int(x)), 2, (0, 255, 0), 3)
    cv2.imwrite("./pic/coordinate/" + str(name_str) + ".jpg", img)

    second_data = np.hstack([str(name_str), g_mean, effective_flame, high_tem_flame, y, x, front_data[:, 1]]).reshape(1, -1)
    print(second_data)
    pd.DataFrame(second_data).to_csv(second_csv, mode='a', index=False, header=False)


if __name__ == '__main__':
    flame, high_temperature = 80, 120  # 火焰, 高温阈值
    dir = "./pic/target"
    img_list = os.listdir(dir)
    img_list.sort(key=lambda x: int(x.replace("target.mp4_", "").split('.')[0]))
    sec_doc_path = './doc/target_10s.csv'
    min_doc_path = './doc/target_mins.csv'
    front_path = './doc/front_10s.csv'

    # 10s pic -> csv
    p = Pool(4)
    for i in img_list:
        p.apply_async(main_function, args=(dir, i, flame, high_temperature, sec_doc_path,))
    p.close()
    p.join()

    # 10s -> 1min
    df = pd.read_csv(sec_doc_path, header=None).values
    df = df[np.lexsort(df[:, ::-1].T)]  # 重排10s
    pd.DataFrame(df).to_csv(sec_doc_path, index=False, header=False)
    gray_col, flame_col = df[:, 1], df[:, 2]

    step = np.where(df[:, 0] % 100 == 0)
    minutes = np.delete(df[step], 0, axis=0)  # drop 0s
    minutes[:, 0] = minutes[:, 0] / 100

    # # 灰度var, 火焰var
    gray_var, effective_var = get_variance(step[0], gray_col), get_variance(step[0], flame_col)
    pd.DataFrame(np.hstack([minutes, effective_var, gray_var])).to_csv(min_doc_path, index=False, header=False)
