from email.header import Header

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import  Axes3D
from matplotlib.projections import projection_registry
from pandas.core.interchange.dataframe_protocol import DataFrame
from sympy.abc import theta

path = 'ex1data2.txt'
data = pd.read_csv(path,header=None,names=['Size','Bedrooms','Price'])

#特征缩放
means = np.array(data.mean().values).reshape(1,3)
std = np.array(data.std().values).reshape(1,3) #保存mean std
print(means.dtype)
temp = data.copy()
data = (data - data.mean(axis=0))/data.std(axis=0) #data.mean(axis=0)为每列替换为平均值
print(data.head())

data.insert(0,'ones',1)
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]  # X是所有行去掉最后一列
Y = data.iloc[:, cols - 1:cols]

X = np.array(X.values)
Y = np.array(Y.values)
theta_begin = np.array([0, 0,0]).reshape(1, 3)  # 调整为(1,3)形状以匹配原矩阵形状

iters=1000
cost=np.zeros(iters)

# 代价函数和梯度下降函数
def cost_function(X, Y, theta):
    inner = np.power((X @ theta.T) - Y, 2)
    return np.sum(inner) / (2 * len(X))
def gradient_descent(X, y, theta, alpha, iters):
    temp = np.zeros(theta.shape)
    parameters = theta.ravel().shape[0]
    global cost

    for i in range(iters):
        error = (X @ theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j].reshape(-1, 1))
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp.copy()
        cost[i] = cost_function(X, Y, theta)

    return theta

g = gradient_descent(X,Y,theta_begin,0.01,iters)
print(g)
print(cost_function(X,Y,g))

# 迭代cost可视化
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('iters')
ax.set_ylabel('cost')
ax.set_title('change of cost')
plt.show()

#把参数转换回来
def theta_back(theta,means,std):
    thetaback = theta.copy()

    thetaback[0,0] = (theta[0,0]-(np.sum((means[0,:-1]*theta[0,1:])/std[0,:-1])))*std[0,-1]+means[0,-1] #目标变量也被标准化

    thetaback[0,1:] = (theta[0,1:]/std[0,:-1])*std[0,-1]
    return thetaback

gf = theta_back(g,means,std)
print(gf)
