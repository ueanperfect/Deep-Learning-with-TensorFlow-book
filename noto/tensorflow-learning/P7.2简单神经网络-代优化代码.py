# noinspection PyUnresolvedReferences

import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from    mpl_toolkits.mplot3d import Axes3D
from    matplotlib import pyplot as plt
import seaborn as sns
from    tensorflow import keras
from    tensorflow.keras import datasets, layers

N_SAMPLES = 2000
TEST_SIZE = 0.3

X, y = sklearn.datasets.make_moons(n_samples=N_SAMPLES, noise=0.2, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
print(X_train.shape, y_train.shape)

# def make_plot(X,y,plot_name,file_name=None,XX=None,YY=None,preds=None,dark=False):
#     if (dark):
#         plt.style.use('dark_background')
#     else:
#         sns.set_style("whitegrid")
#     plt.figure(figsize=(16,32))
#     axes = plt.gca()
#     axes.set(xlabel="$x_1$",ylabel = "$x_2$")
#     plt.title(plot_name,fontsize = 30)
#     plt.subplots_adjust(left=0.2)
#     plt.subplots_adjust(right=0.8)
    # if (XX is not None and YY is not None and preds is not None):
    #     plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha=1, cmap=cm.Spectral)
    # plt.contour(XX, YY, preds.reshape(XX.shape), levels=[.5], cmap="Greys", vmin=0, vmax=.6)
    # 绘制散点图，根据标签区分颜色
    # plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=40, cmap=plt.cm.Spectral, edgecolors='none')
    # plt.savefig('dataset.svg')
    # plt.close()
    # # 调用 make_plot 函数绘制数据的分布，其中 X 为 2D 坐标，y 为标签
    # make_plot(X, y, "Classification Dataset Visualization ")
    # plt.show()
#make_plot(X,y,"ok")
class Layer:
    def __init__(self,n_input,n_neurons,activation = None,weights = None,bias=None):
        self.weights = weights if weights is not None else np.random.randn(n_input, n_neurons) * np.sqrt(1 / n_neurons)
        self.bias = bias if bias is not None else np.random.rand(n_neurons) *0.1
        self.activation = activation  # 激活函数类型，如’sigmoid’
        self.last_activation = None # 激活函数的输出值 o
        self.error = None  # 用于计算当前层的 delta 变量的中间变量
        self.delta = None # 记录当前层的 delta 变量，用于计算梯度

    def activate(self, x):
        # 前向传播函数
        r = np.dot(x,self.weights) + self.bias  # X@W+b # 通过激活函数，得到全连接层的输出 o self.last_activation = self._apply_activation(r)
        return self.last_activation

    def _apply_activation(self, r):  # 计算激活函数的输出
        if self.activation is None:
            return r  # 无激活函数，直接返回
        # ReLU激活函数
        elif self.activation == 'relu':
            return np.maximum(r, 0)  # tanh激活函数

        elif self.activation == 'tanh':
            return np.tanh(r)
        # sigmoid激活函数
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))
        return r

    def apply_activation_derivative(self, r):
        if self.activation is None:
            return np.ones_like(r)
        elif self.activation == 'relu':
            grad = np.array(r, copy=True)
            grad[r > 0] = 1.
            grad[r <= 0] = 0.
            return grad
        elif self.activation == 'tanh':
            return 1 - r ** 2  # Sigmoid函数的导数实现
        elif self.activation == 'sigmoid':
            return r * (1 - r)
        return r

class NeuralNetwork:
    # 神经网络模型大类
    def __init__(self):
        self._layers = [] # 网络层对象列表
    def add_layer(self, layer): # 追加网络层
        self._layers.append(layer)
    def feed_forward(self, X): # 前向传播
        for layer in self._layers: # 依次通过各个网络层
            X = layer.activate(X)
        return X

nn = NeuralNetwork() # 实例化网络类
nn.add_layer(Layer(2, 25, 'sigmoid')) # 隐藏层1, 2=>25
nn.add_layer(Layer(25, 50, 'sigmoid')) # 隐藏层2, 25=>50

nn.add_layer(Layer(50, 25, 'sigmoid'))  # 隐藏层3, 50=>25
nn.add_layer(Layer(25, 2, 'sigmoid')) # 输出层, 25=>2

def backpropagation(self, X, y, learning_rate): # 反向传播算法实现
    # 前向计算，得到输出值
    output = self.feed_forward(X)
    for i in reversed(range(len(self._layers))): # 反向循环
        layer = self._layers[i] # 得到当前层对象
    # 如果是输出层
        if layer == self._layers[-1]: # 对于输出层
            layer.error = y - output # 计算 2 分类任务的均方差的导数 # 关键步骤:计算最后一层的 delta，参考输出层的梯度公式
            layer.delta = layer.error *layer.apply_activation_derivative(output)
        else: # 如果是隐藏层
            next_layer = self._layers[i + 1] # 得到下一层对象
            layer.error = np.dot(next_layer.weights, next_layer.delta) # 关键步骤:计算隐藏层的 delta，参考隐藏层的梯度公式
            layer.delta = layer.error *layer.apply_activation_derivative(layer.last_activation)

            def backpropagation(self, X, y, learning_rate): # 代码接上面
                # 循环更新权值
                for i in range(len(self._layers)):
                    layer = self._layers[i]
                # o_i为上一网络层的输出
                    o_i = np.atleast_2d(X if i == 0 else self._layers[i -1].last_activation)
                # 梯度下降算法，delta 是公式中的负数，故这里用加号
                    layer.weights += layer.delta * o_i.T * learning_rate

def train(self, X_train, X_test, y_train, y_test, learning_rate,max_epochs):
    y_onehot = np.zeros((y_train.shape[0], 2))
    y_onehot[np.arange(y_train.shape[0]), y_train] = 1

    mses = []
    for i in range(max_epochs):  # 训练1000个epoch
        for j in range(len(X_train)):  # 一次训练一个样本
            self.backpropagation(X_train[j], y_onehot[j], learning_rate)
        if i % 10 == 0:
    # 打印出 MSE Loss
            mse = np.mean(np.square(y_onehot - self.feed_forward(X_train)))
            mses.append(mse)
            print('Epoch: #%s, MSE: %f' % (i, float(mse)))
    # 统计并打印准确率
            print('Accuracy: %.2f%%' % (self.accuracy(self.predict(X_test), y_test.flatten()) * 100))
    return mses