#coding:utf8
from numpy import *
import matplotlib.pyplot as plt

# 用于生成随机的样本点
def randPoint():
    x = []
    y = []
    for i in range(130):
        a = random.uniform(-1000, 1000)
        b = random.uniform(-1000, 1000)
        x.append(a)
        y.append(b)
    f = open('D:\\PycharmProjects\\Algorithm\\kmeans\\point.txt', 'w')
    for i, j in zip(x, y):
        f.write((i, j).__str__()+'\n')
    f.close()
    plt.xlim(-1000, 1000)
    plt.ylim(-1000, 1000)
    plt.scatter(x, y)
    plt.show()
    
if __name__  ==  '__main__':
     #randPoint()
    