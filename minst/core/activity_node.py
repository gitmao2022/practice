'''
@Description  : file content
@Version      : 1.0
@Author       : gitmao2022
@Date         : 2025-03-23 22:45:59
@LastEditors  : gitmao2022
@LastEditTime : 2026-02-04 20:49:48
@FilePath     : activity_node.py
@Copyright (C) 2025  by ${gitmao2022}. All rights reserved.
'''


import numpy as np
from .node import Node


class Logistic(Node):
    """
    对向量的分量施加Logistic函数
    """

    def compute_value(self):
        x = self.parents[0].value
        # 对父节点的每个分量施加Logistic
        return 1.0 / (1.0 + np.power(np.e, np.where(-x > 1e2, 1e2, -x)))
    
    def get_jacobi(self, parent):
        return np.diag(np.multiply(self.value, 1 - self.value).flatten())


class ReLU(Node):
    """
    对矩阵的元素施加ReLU函数
    """

    nslope = 0.1  # 负半轴的斜率

    def compute_value(self):
        return np.mat(np.where(
            self.parents[0].value > 0.0,
            self.parents[0].value,
            self.nslope * self.parents[0].value)
        )

    def get_jacobi(self, parent):
        return np.diag(np.where(self.parents[0].value.A1 > 0.0, 1.0, self.nslope))


class SoftMax(Node):
    """
    SoftMax函数
    """

    def compute_value(self):
        #if parents[0].value.ndim ==2 ,则按行计算SoftMax
        x = self.parents[0].value
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)


    def get_jacobi(self, parent):
        """
        计算SoftMax函数的雅可比矩阵,要考虑parent的维度为2的情况
        """
        s = self.value
        if s.ndim == 1:
            s = s.reshape(-1, 1)
        jacobi = np.zeros((s.shape[0], s.shape[1], s.shape[1]))
        for i in range(s.shape[0]):
            for j in range(s.shape[1]):
                for k in range(s.shape[1]):
                    if j == k:
                        jacobi[i, j, k] = s[i, j] * (1 - s[i, k])
                    else:
                        jacobi[i, j, k] = -s[i, j] * s[i, k]
        return jacobi
        

class Step(Node):
    
    def compute_value(self):
        return np.where(self.parents[0].value >= 0.0, 1.0, 0.0)

    def get_jacobi(self, parent):
        return np.zeros(self.dimension())