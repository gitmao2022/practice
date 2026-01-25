'''
@Description  : 创建神经网络节点节点
@Version      : 1.0
@Author       : gitmao2022
@Date         : 2025-02-15 21:15:34
@LastEditors  : gitmao2022
@LastEditTime : 2026-01-25 16:48:55
@FilePath     : node.py
@Copyright (C) 2025  by ${gimao2022}. All rights reserved.
'''

import numpy as np
from abc import abstractmethod
from .graph import default_graph


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
        self.graph = kargs.get('graph', default_graph)
        self.need_save = kargs.get('need_save', True)
        self.node_name = kargs.get('node_name', '{}:{}'.format( 
            self.__class__.__name__, self.graph.node_count()))
        # 将本节点添加到父节点的子节点列表中 
        for parent in self.parents:
            parent.children.append(self)
        # 将本节点添加到计算图中
        self.graph.add_node(self)

    def set_value(self, value, clear=True):
        """
        设置节点的值
        """
        if clear:
            self.clear_value()
        self.value = value

    def get_value(self):
        """
        获取节点的值
        """
        return self.value
    
    @abstractmethod
    def get_jacobi(self, parent):
        """
        抽象方法，计算本节点对某个父节点的雅可比矩阵
        """

    def clear_jacobi(self):
        """
        清空结果节点对本节点的雅可比矩阵
        """
        self.jacobi = None

    def forward(self):
        for node in self.parents:
            if node.value is None:
                node.forward()
        self.set_value(self.compute_value())
    
    @property
    def shape(self):
        """
        返回节点值的形状
        """
        return self.value.shape
    
    
    def clear_value(self):
        for child in self.children:
            child.clear_value()
        self.value = None

    def compute_value(self):
        return self.value
        

    def dimension(self):
        """
        返回本节点的值展平成向量后的维数,不限于二维向量
        """
        return np.prod(self.shape)
    
    def backward(self, result):
        """
        反向传播，计算结果节点对本节点的雅可比矩阵
        """
        if self.jacobi is None:
            if self is result:
                self.jacobi = np.eye(self.dimension())
            else:
                self.jacobi = np.zeros((result.dimension(), self.dimension()))
                for child in self.children:
                    if child.value is not None:
                        self.jacobi +=np.dot(child.backward(result), child.get_jacobi(self))
        return self.jacobi
    
