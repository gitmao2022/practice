'''
@Description  : 用于numpy操作的辅助函数或者类
@Version      : 1.0
@Author       : gitmao2022
@Date         : 2023-03-10 21:44:51
@LastEditors  : gitmao2022
@LastEditTime : 2023-05-24 15:21:54
@FilePath     : npas.py
@Copyright (C) 2023  by gimao2022. All rights reserved.
'''

import numpy as np


def linerreg_feature_scaling(X, modle='min-max'):
    '''
    @description: 对数据进行特征缩放，适用于线性回归
    @param X {拟进行缩放的数据，numpy格式，数组维度不超过2}: 
    @param modle {'min-max':min-max归一化;'test':在该测试中，
    每个数据都除以所有数据的最大值减去最小值，也即每个数据都进行了相同的线性变换；
    'standard':标准化处理} 
    @return 返回缩放后的numpy（不改变原X值）
    '''
    if modle == 'min-max':
        Ans = X.astype(np.float16)
        for i in range(len(X[0])):
            l_max, l_min = np.max(X[:, i]), np.min(X[:, i])
            c = l_max - l_min
            #print('c=',c)
            Ans[:, i] = np.around((X[:, i] - l_min) / c, 2) if c != 0 else 0

    if modle == 'test':
        Ans = X.astype(np.float16)
        diff = np.max(X)-np.min(X)
        Ans = Ans / diff

    if modle=='standard':
        Ans = X.astype(np.float16)
        mean = np.mean(Ans, axis=0)
        std = np.std(Ans, axis=0)
        Ans = (Ans - mean) / std 
       # Ans = np.where(std != 0, (Ans - mean) / std, 0)
        Ans=np.where(Ans!=0,Ans,1)

    return Ans


def add_right_ones(X):
    '''
    @为numpy数据最右侧添加一列1
    @return：返回修改后的numpy数据
    '''
    L = np.ones(shape=[len(X)], dtype=X.dtype)
    return np.column_stack((X, L))

# 定义sigmoid函数及其导数
def sigmoid(X):
    X[X>0]=1 / (1 + np.exp(-X))
    X[X<0]=np.exp(X) / (1 + np.exp(X))

def sigmoid_derivative(X):
    return sigmoid(X) * (1 - sigmoid(X))


def softmax(X,axis=1):
    row_max = np.max(X, axis=axis).reshape(-1, 1)
    X -= row_max
    X_exp = np.exp(X)
    return X_exp / np.sum(X_exp, axis=axis, keepdims=True)

# 定义Relu函数及其导数
def Relu(x):
    return np.maximum(0, x)

def Relu_derivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x