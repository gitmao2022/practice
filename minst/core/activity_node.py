'''
@Description  : file content
@Version      : 1.0
@Author       : gitmao2022
@Date         : 2025-03-23 22:45:59
@LastEditors  : gitmao2022
@LastEditTime : 2025-04-29 21:23:19
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

    @staticmethod
    def softmax(a):
        a[a > 1e2] = 1e2  # 防止指数过大
        ep = np.power(np.e, a)
        return ep / np.sum(ep)

    def compute_value(self):
        return SoftMax.softmax(self.parents[0].value)

    def get_jacobi(self, parent):
        """
        我们不实现SoftMax节点的get_jacobi函数，
        训练时使用CrossEntropyWithSoftMax节点
        """
        raise NotImplementedError("Don't use SoftMax's get_jacobi")
    

class Step(Node):
    
    def compute_value(self):
        return np.where(self.parents[0].value >= 0.0, 1.0, 0.0)

    def get_jacobi(self, parent):
        return np.zeros(self.dimension())