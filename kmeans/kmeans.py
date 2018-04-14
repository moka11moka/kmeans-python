# coding:utf8
# 实现kmeans聚类算法

from numpy import *
import matplotlib.pyplot as plt
import copy
# 利用numpy的random生成随机的一个1000数组
# 然后描点


def describe_point():
    a = random.randn(1000)
    b = [i*1000 for i in a]
    copy.deepcopy(b)
    random.shuffle(b)
    f = open('D:\\PycharmProjects\\Algorithm\\kmeans\\point.txt', 'w')
    for i, j in zip(a, b):
        f.write((i, j).__str__()+'\n')
    plt.plot(a, b, 'ro')
    plt.show()

# 将文本中的点加载到一个矩阵中
def loadDateSet(fileName):
    dataMat = []
    f = open(fileName, 'r')
    for line in f.readlines():
        curLine = line.strip().split(',')
        resultSet = [float(i) for i in curLine]
        dataMat.append(resultSet)
    return dataMat


# 在样本中随机取k个点作为初始质心

def initCentroids(dataSet, k):
    # 矩阵的行数、列数
    num, dim = dataSet.shape
    centroids = zeros((k, dim))
    for i in range(k):
        index = int(random.uniform(0, num))
        centroids[i, :] = dataSet[index, :]
    return centroids

# 欧拉距离
def euclDistance(vector1, vector2):
    return sqrt(sum(power(vector2 - vector1, 2)))


def kmeans(dataSet, k, distant=euclDistance, createCent=initCentroids):
    # 得到样本数
    m = shape(dataSet)[0]
    # 初始化一个m*2的矩阵, 用于存放点所属质心，和距离
    cluster = mat(zeros((m, 2)))
    # 初始化k个质心
    centerPoint = createCent(dataSet, k)
    # 用于判断是否确定所属的质心
    isTrue = True
    while isTrue:
        isTrue = False
        for i in range(m):
            min_dis = inf; min_index = -1
            for j in range(k):
                # 求出每一个质心点到样本点的欧拉距离，最后将距离最小的放入到数组cluster中
                disIJ = distant(dataSet[i, :], centerPoint[j, :])
                if disIJ < min_dis:
                    min_dis = disIJ; min_index = j
            if cluster[i, :0] != min_dis: isTrue = True
            cluster[i, :] = min_index, min_dis**2

    # 用于求出某个质心点下的所有样本点的平均值，赋值为新的质心点
    for cent in range(k):
        pointCluster = dataSet[nonzero(cluster[:, 0].A==cent)[0]]
        centerPoint[cent, :] = mean(pointCluster, axis=0)
    return cluster, centerPoint

# 画图，将点显示在平面中
def draw(dataSet, center):
    center_num = len(center)
    fig = plt.figure
    # 在图上描点
    plt.xlim(-1000, 1000)
    plt.ylim(-1000, 1000)
    plt.scatter(dataSet[:, 0], dataSet[:, 1])
    for i in range(center_num):
        # 给箭头符号做注释
        plt.annotate('center', xy=(center[i, 0], center[i, 1]), xytext= \
            (center[i, 0] + 1, center[i, 1] + 1), arrowprops=dict(facecolor='blue'))
    plt.show()


# 测试

if __name__ == '__main__':
    dataMat = mat(loadDateSet('D:\\PycharmProjects\\Algorithm\\kmeans\\point.txt'))
    dataSet, center = kmeans(dataMat, 6)
    draw(dataMat, center)