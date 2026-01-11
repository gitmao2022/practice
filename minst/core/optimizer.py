'''
@Description  : file content
@Version      : 1.0
@Author       : gitmao2022
@Date         : 2025-10-02 15:09:15
@LastEditors  : gitmao2022
@LastEditTime : 2026-01-01 14:08:36
@FilePath     : optimizer.py
@Copyright (C) 2025  by ${gitmao2022}. All rights reserved.
'''

from .graph import *
from .node import *
from .operate_node import *
from .loss_node import *
from .activity_node import *
from .variable_node import *
import numpy as np 

class Optimizer:   

    def gnr_batch_var(self):
        self.batch_no=np.random.choice(self.train_set.shape[0], self.batch_size, replace=False)
        self.input_var.set_value(self.train_set[self.batch_no,:])
        self.target_var.set_value(self.target_set[self.batch_no,:])
        
    def __init__(self, epoch,batch_size,train_set,target_set,
                 learning_rate=0.01,optimizer_type='adam'):
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.train_set = train_set
        self.target_set = target_set
        self.jacobi_cache={}
        #train或者target集合可能只有一个维度，所以reshape一下
        if len(self.train_set.shape)==1:
            self.train_set = self.train_set.reshape((-1,1))
        if len(self.target_set.shape)==1:
            self.target_set = self.target_set.reshape((-1,1))
        # 确保第一个维度为 batch_size，后续维度与数据的其余维度一致
        input_dim = (self.batch_size,) + tuple(self.train_set.shape[1:])
        target_dim = (self.batch_size,) + tuple(self.target_set.shape[1:])
        self.input_var = Variable(dim=input_dim, init=False, trainable=False)
        self.target_var = Variable(dim=target_dim, init=False, trainable=False)
        self.gnr_batch_var()

    def forward_backward(self,epoch=1):
        """
        前向传播计算结果节点的值并反向传播计算结果节点对各个节点的雅可比矩阵
        """
        for _ in range(epoch):
            default_graph.clear_jacobi()
            self.jacobi_cache={}
            #重新生成batch数据
            self.gnr_batch_var()
            self.forward()
            # default_graph.draw()
            for node in default_graph.nodes:
                if isinstance(node, Variable) and node.trainable and self.jacobi_cache.get(node.node_name) is None:
                    node.backward(self.loss_node)
                    jacobi_mean=np.mean(node.jacobi,axis=0).reshape(node.shape)        
                    self.jacobi_cache[node.node_name]=jacobi_mean
            # print(self.jacobi_cache)        
            for node in default_graph.nodes:
                if isinstance(node, Variable) and node.trainable:
                    jacobi_mean=self.jacobi_cache[node.node_name]
                    node.set_value(node.value - self.learning_rate * jacobi_mean)

           
    def forward(self):
        """
        前向传播计算结果节点的值
        """
        # default_graph.clear_jacobi()
        # default_graph.clear_changeable_value()
        self.loss_node.forward()
      
    def add_fc_layer(self,previous_layer, back_layer_size, activation):
        """
        :param previous_layer: 输入向量
        :param back_layer_size: 输出向量的维度
        :param activation: 激活函数类型
        :return: 输出向量
        """
        first_layer_size = previous_layer.value.shape[1]
        weights = Variable((first_layer_size, back_layer_size), init=True, trainable=True)
        bias = Variable((1, back_layer_size), init=True, trainable=True)
        affine = Add(MatMul(previous_layer, weights), bias)
        if activation == "ReLU":
            return ReLU(affine)
        elif activation == "Logistic":
            return Logistic(affine)
        else:
            return affine




