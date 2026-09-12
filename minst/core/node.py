'''
@Description  : Create neural network nodes
@Version      : 1.0
@Author       : gitmao2022
@Date         : 2025-02-15 21:15:34
@LastEditors  : gitmao2022
@LastEditTime : 2026-09-08 17:04:17
@FilePath     : node.py
@Copyright (C) 2025  by ${gimao2022}. All rights reserved.
'''

import numpy as np
from abc import abstractmethod
from .graph import default_graph



class Node(object):
    """
    Base class for nodes.
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
        # Add this node to each parent's list of children.
        for parent in self.parents:
            parent.children.append(self)
        # Add this node to the computation graph.
        self.graph.add_node(self)

    def set_value(self, value, clear=True):
        """
        Set the node's value.
        """
        # Node values are not restricted to three dimensions.
        if not isinstance(value, np.ndarray):
            value = np.asarray(value)
        if clear:
            self.clear_value()
        self.value = value

    def get_value(self):
        """
        Get the node's value.
        """
        return self.value
    
    @abstractmethod
    def get_jacobi(self, parent):
        """
        抽象方法，计算本节点对某个父节点的雅可比矩阵
        如果parent的维度大于1，则无法计算jacobi矩阵，所以要将parent维度延展至1维
        """

    def clear_jacobi(self):
        """
        Clear the Jacobian of the result node with respect to this node.
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
        Return the shape of the node's value.
        """
        return self.value.shape
    
    
    def clear_value(self):
        for child in self.children:
            child.clear_value()
        self.value = None

    def compute_value(self):
        return self.value

    def backward_jacobi_product(self, output_jacobi, parent):
        return None
        

    def dimension(self):
        """
        Return the dimension of the node's flattened value, including values beyond two dimensions.
        """
        return np.prod(self.shape)
    
    def backward(self, result):
        """
        Perform backpropagation and compute the result node's Jacobian with respect to this node.
        """
        if self.jacobi is None:
            if self is result:
                self.jacobi = np.eye(self.dimension())

            else:
                self.jacobi = np.zeros((result.dimension(), self.dimension()))
                for child in self.children:
                    if child.value is not None:
                        #catch ValueError exception when shapes are not aligned
                        try:
                            child_jacobi = child.backward(result)
                            product = child.backward_jacobi_product(child_jacobi, self)
                            if product is None:
                                product = np.dot(child_jacobi, child.get_jacobi(self))
                            self.jacobi += product
                        except ValueError as e:
                            print(f"ValueError in backward propagation at node {child.node_name}:{e}")  
                            print('self.node_name', self.node_name,'self.shape', self.shape)
                            print('child.node_name', child.node_name,'child.shape', child.shape)
                            print("child.get_jacobi(self).shape",child.get_jacobi(self).shape)
                            print("child.backward(result).shape",child.backward(result).shape)
                            exit(1)                      
        return self.jacobi
    
