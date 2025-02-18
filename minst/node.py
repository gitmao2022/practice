'''
@Description  : 创建神经网络节点节点
@Version      : 1.0
@Author       : gitmao2022
@Date         : 2025-02-15 21:15:34
@LastEditors  : gitmao2022
@LastEditTime : 2025-02-18 21:46:04
@FilePath     : node.py
@Copyright (C) 2025  by ${gimao2022}. All rights reserved.
'''

import abc
import numpy as np  

class Node(object):
    """
    节点基类
    """
    def __init__(self, *parents, **kargs):
        
        self.kargs = kargs
        self.parents = list(parents) 
        self.value = None
        self.jacobi = None
        self.children = []

    def set_value(self, value):
        """
        设置节点的值
        """
        self.value = value

    def get_value(self):
        """
        获取节点的值
        """
        return self.value

    def add_child(self, child):
        """
        添加子节点
        """
        self.children.append(child)

    def get_children(self):
        """
        获取子节点
        """
        return self.children

    def forward(self):
        """
        前向传播，计算节点的值
        """
        raise NotImplementedError()

    def backward(self, result):
        """
        反向传播，计算结果节点对本节点的雅可比矩阵
        """
        if self.jacobi is None:
            if self is result:
                self.jacobi = np.mat(np.eye(self.dimension()))
            else:
                self.jacobi = np.mat(
                    np.zeros((result.dimension(), self.dimension())))

                for child in self.get_children():
                    if child.value is not None:
                        self.jacobi += child.backward(result) * child.get_jacobi(self)