'''
@Description  : file content
@Version      : 1.0
@Author       : gitmao2022
@Date         : 2025-03-23 22:45:59
@LastEditors  : gitmao2022
@LastEditTime : 2026-02-22 13:32:25
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
        return np.diag(np.where(self.parents[0].value.flatten() > 0.0, 1.0, self.nslope))

        
class Softmax(Node):
    """
    对矩阵的行施加Softmax函数
    """
    def compute_value(self):
        x = self.parents[0].value
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def get_jacobi(self, parent):
        # 这里存在重复计算，但为了代码清晰简洁，舍弃进一步优化
        print("Softmax节点的雅可比矩阵计算存在性能问题,故返回0作为占位 。")
        return 0
    

class Step(Node):
    
    def compute_value(self):
        return np.where(self.parents[0].value >= 0.0, 1.0, 0.0)

    def get_jacobi(self, parent):
        return np.zeros(self.dimension())