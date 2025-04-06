import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.core.interchange.dataframe_protocol import DataFrame
from sympy.abc import theta

path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

# 绘制散点图
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))


# plt.show()

# 代价函数
def cost_function(X, Y, theta):
    inner = np.power((X @ theta.T) - Y, 2)
    return np.sum(inner) / (2 * len(X))


# 准备数据
data.insert(0, 'ones', 1)
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]  # X是所有行去掉最后一列
Y = data.iloc[:, cols - 1:cols]

X = np.array(X.values)
Y = np.array(Y.values)
theta_begin = np.array([0, 0]).reshape(1, 2)  # 调整为(1,2)形状以匹配原矩阵形状
print(X.shape)
print(theta_begin.shape)
print(Y.shape)

# 梯度下降
iters = 1000
cost = np.zeros(iters)


def gradientDescent(X, y, theta, alpha, iters):
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


g = gradientDescent(X, Y, theta_begin, 0.01, 1000)
print(g)
print(cost[-1])

# 绘制结果
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

# 绘制代价函数变化曲线（可选）
fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.plot(np.arange(iters), cost, 'r')
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Cost')
ax2.set_title('cost_change')
plt.show()
