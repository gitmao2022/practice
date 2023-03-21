'''
@Description  : 用于numpy操作的辅助函数或者类
@Version      : 1.0
@Author       : gitmao2022
@Date         : 2023-03-10 21:44:51
@LastEditors  : gitmao2022
@LastEditTime : 2023-03-21 22:32:29
@FilePath     : npas.py
@Copyright (C) 2023  by gimao2022. All rights reserved.
'''

import numpy as np


def linerreg_feature_scaling(X, modle='min-max'):
    '''
    @description: 对数据进行特征缩放，适用于线性回归
    @param X {拟进行缩放的数据，numpy格式，数组维度不超过2}: 
    @param modle {'min-max':min-max归一化;'test':在该测试中，每列数据
    除以该列的最大值，使得数据均在0-1间，并且使每列数据进行了相同参数的线性变换}: 
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
        Max_var = np.max(X, axis=0)
        Ans = Ans / Max_var

    return Ans


def add_right_ones(X):
    '''
    @为numpy数据最右侧添加一列1
    @return：返回修改后的numpy数据
    '''
    L = np.ones(shape=[len(X)], dtype=X.dtype)
    return np.column_stack((X, L))
