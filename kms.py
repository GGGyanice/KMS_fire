# -*- coding:utf-8 -*-
"""
Time : 2020/10/30 8:56
Author : Kexin Guan
Decs ：
"""
import os
import shutil
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def distE(vec_a, vec_b):
    return np.sqrt(sum(np.power(vec_a-vec_b, 2)))  # 欧式距离


# 随机选择中心点
def rand_cent(dataset, k):
    centroids = np.zeros((k, dataset.shape[1]))
    for i in range(k):
        index = int(np.random.uniform(0, dataset.shape[0]))
        centroids[i, :] = dataset[index, :]  # 质心
    return centroids


def kmeans(dataset, k, dis_mea=distE, create_cent=rand_cent):
    """

    :param dataset: rect.csv
    :param k: 4
    :param dis_mea: 欧氏距离
    :param create_cent: 随机质心, shape: (k, 30)
    :return: 质心（k=4, dataset.shape[1]:对应维的算术平方值）
            聚类结果（簇索引值: k, 误差）
    """
    # frist column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    cluster_assment = np.mat(np.zeros((dataset.shape[0], 2)))
    cluster_changed = True

    # init centroids
    centroids = create_cent(dataset, k)

    while cluster_changed:
        cluster_changed = False
        
        for i in range(dataset.shape[0]):
            min_dist, min_index = np.inf, -1
            # for each centroid
            # find the centroid who is closest
            for j in range(k):
                distance = dis_mea(centroids[j, :], dataset[i, :])
                if distance < min_dist:
                    min_dist = distance
                    min_index = j

            # update its cluster
            if cluster_assment[i, 0] != min_index:
                cluster_changed = True  # 继续聚类
                cluster_assment[i, :] = min_index, min_dist ** 2

        # update centroids
        for j in range(k):
            points_in_cluster = dataset[np.nonzero(cluster_assment[:, 0].A == j)[0]]  # 取非0列号为点群
            centroids[j, :] = np.mean(points_in_cluster, axis=0)
    return centroids, cluster_assment


def show_result(data_set, k, centriods, clusterA):
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    for i in range(data_set.shape[0]):
        index = int(clusterA[i, 0])
        plt.plot(data_set[i, 0], data_set[i, 1], mark[index])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^r', '+b', 'sb', 'db', '<b', 'pb']
    for i in range(k):
        plt.plot(centriods[i, 0], centriods[i, 1], mark[i], markersize=12)
    plt.show()


if __name__ == '__main__':
    rect_csv = './doc/rect.csv'
    centroid_csv = "./doc/myCentroids.csv"
    clust_csv = "./doc/clustAssing.csv"
    dir = "./pic/target"
    img_list = os.listdir(dir)
    img_list.sort(key=lambda x: int(x.replace("target.mp4_", "").split('.')[0]))

    print("---- strat KMS and draw ----")
    k = 4
    data = pd.read_csv(rect_csv).values[:, 1:]
    # pcaClf = PCA(n_components=2, whiten=True)
    # pcaClf.fit(data)
    # data = pcaClf.transform(data)  # 降低维度 展示效果
    my_centroids, clust_assing = kmeans(data, k)
    show_result(data, k, my_centroids, clust_assing)

    pd.DataFrame(my_centroids).to_csv(centroid_csv, index=False, header=False)
    pd.DataFrame(clust_assing).to_csv(clust_csv, index=False, header=False)

    print("---- remove pic ----")
    label = pd.read_csv(clust_csv, header=None).values[:, 0]
    for l in range(len(img_list)):
        img_path = os.path.join(dir, img_list[l])
        name_str = img_path[-14:-8]
        print(img_path, label[l])
        shutil.copy(img_path, './pic/kms/k=' + str(k)+'/' + str(int(label[l])) + '/' + str(name_str) + '.jpg')
