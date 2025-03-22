# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:23:01 2019

@author: zhangjuefei
"""
import abc

import numpy as np

from .graph import Graph, default_graph


class Node(object):
    """
    计算图节点类基类
    """

    def __init__(self, *parents, **kargs):

        # 计算图对象，默认为全局对象default_graph
        self.kargs = kargs
        self.graph = kargs.get('graph', default_graph)
        self.need_save = kargs.get('need_save', True)
        self.node_name=kargs.get('node_name','{}:{}'.format(
            self.__class__.__name__, self.graph.node_count()))

        self.parents = list(parents)  # 父节点列表
        self.children = []  # 子节点列表
        self.value = None  # 本节点的值
        self.jacobi = None  # 结果节点对本节点的雅可比矩阵

        # 将本节点添加到父节点的子节点列表中
        for parent in self.parents:
            parent.children.append(self)

        # 将本节点添加到计算图中
        self.graph.add_node(self)

    def forward(self):
        """
        前向传播计算本节点的值，若父节点的值未被计算，则递归调用父节点的forward方法
        """
        for node in self.parents:
            if node.value is None:
                node.forward()

        self.compute()

    @abc.abstractmethod
    def compute(self):
        """
        抽象方法，根据父节点的值计算本节点的值
        """

    @abc.abstractmethod
    def get_jacobi(self, parent):
        """
        抽象方法，计算本节点对某个父节点的雅可比矩阵
        """

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

        return self.jacobi

    def clear_jacobi(self):
        """
        清空结果节点对本节点的雅可比矩阵
        """
        self.jacobi = None

    def dimension(self):
        """
        返回本节点的值展平成向量后的维数
        """
        return self.value.shape[0] * self.value.shape[1]

    def shape(self):
        """
        返回本节点的值作为矩阵的形状：（行数，列数）
        """
        return self.value.shape

    def reset_value(self, recursive=True):
        """
        重置本节点的值，并递归重置本节点的下游节点的值
        """

        self.value = None

        if recursive:
            for child in self.children:
                child.reset_value()


class Variable(Node):
    """
    变量节点
    """

    def __init__(self, dim, init=False, trainable=True, **kargs):
        """
        变量节点没有父节点，构造函数接受变量的形状，是否初始化以及是否参与训练的标识
        """
        Node.__init__(self,  **kargs)
        self.dim = dim

        # 如果需要初始化，则以正态分布随机初始化变量的值
        if init:
            self.value = np.mat(np.random.normal(0, 0.001, self.dim))

        # 变量节点是否参与训练
        self.trainable = trainable

    def set_value(self, value):
        """
        为变量赋值
        """
        assert isinstance(value, np.matrix) and value.shape == self.dim

        # 本节点的值被改变，重置所有下游节点的值
        #self.reset_value()
        self.value = value
