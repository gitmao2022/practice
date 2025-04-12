'''
@Description  : 创建神经网络节点节点
@Version      : 1.0
@Author       : gitmao2022
@Date         : 2025-02-15 21:15:34
@LastEditors  : gitmao2022
@LastEditTime : 2025-04-12 21:30:44
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
        """
        前向传播，计算节点的值
        """
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
    
    def clear_value(self,recursive=True):
        self.value = None
        if recursive:
            for child in self.children:
                child.clear_value()

    @abstractmethod
    def compute_value(self):
        """
        计算节点的值
        """
        pass

    def dimension(self):
        """
        返回本节点的值展平成向量后的维数
        """
        return self.shape[0] * self.shape[1]
    
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
    
