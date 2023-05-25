'''
@Description  : 手写数字识别，用chatgpt生成
@Version      : 1.0
@Author       : chatgpt
@Date         : 2023-05-23 14:56:53
@LastEditors  : gitmao2022
@LastEditTime : 2023-05-24 16:10:25
@FilePath     : test.py
@Copyright (C) 2023  by ${git_name}. All rights reserved.
'''
import sys
sys.path.append("..")
import npas
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# 定义softmax函数及其导数
def softmax(x):
    x -= np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def softmax_derivative(x):
    return softmax(x) * (1 - softmax(x))

# 定义神经网络类
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重矩阵
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)
        
    def forward(self, input_data):
        # 前向传播
        self.z2 = np.dot(input_data, self.weights1)
        self.a2 = npas.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.weights2)
        self.output = softmax(self.z3)
        
    def backward(self, input_data, target_output, learning_rate):
        # 反向传播
        error = target_output - self.output
        delta_output = error * softmax_derivative(self.z3)
        error_hidden = np.dot(delta_output, self.weights2.T)
        delta_hidden = error_hidden * npas.sigmoid_derivative(self.z2)
        
        # 更新权重矩阵
        self.weights2 += learning_rate * np.dot(self.a2.T, delta_output)
        self.weights1 += learning_rate * np.dot(input_data.T, delta_hidden)
        
    def train(self, input_data, target_output, learning_rate, epochs):
        for i in range(epochs):
            self.forward(input_data)
            self.backward(input_data, target_output, learning_rate)
            print('weights1=',self.weights1,'weights2=',self.weights2)
            
    def predict(self, input_data):
        self.forward(input_data)
        return self.output

# 加载手写数字数据集
digits = load_digits()

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# 将目标输出转换为one-hot编码
target_train = np.zeros((y_train.size, y_train.max()+1))
target_train[np.arange(y_train.size), y_train] = 1

target_test = np.zeros((y_test.size, y_test.max()+1))
target_test[np.arange(y_test.size), y_test] = 1

# 创建神经网络对象并进行训练
nn = NeuralNetwork(X_train.shape[1], 10, target_train.shape[1])
nn.train(X_train, target_train, 0.1, 5000)

# 预测测试集的输出结果
#print(X_test)
output = nn.predict(X_test)

'''
# 计算交叉熵损失函数
loss = -np.sum(target_test * np.log(output)) / target_test.shape[0]
print("交叉熵损失函数：", loss)

'''
#print('target-test=',target_test[1:10])
#print('output=',output[1:10])
