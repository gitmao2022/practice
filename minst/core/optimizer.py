'''
@Description  : file content
@Version      : 1.0
@Author       : gitmao2022
@Date         : 2025-10-02 15:09:15
@LastEditors  : gitmao2022
@LastEditTime : 2025-10-02 18:59:40
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
    def __init__(self, epoch,sample_size,learning_rate=0.01,optimizer_type='adam'):
        self.epoch = epoch
        self.sample_size = sample_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
    
    def add_fc_layer(self,previous_layer, back_layer_size, activation,expant_dim=False):
        """
        :param input: 输入向量
        :param input_size: 输入向量的维度
        :param size: 神经元个数，即输出个数（输出向量的维度）
        :param activation: 激活函数类型
        :return: 输出向量
        """
        first_layer_size = previous_layer.value.shape[1]
        if expant_dim:
            input = previous_layer.reshape((sample_size,first_layer_size))
        weights = Variable((first_layer_size, back_layer_size), init=True, trainable=True)
        bias = Variable((1, back_layer_size), init=True, trainable=True)
        affine = Add(MatMul(previous_layer, weights), bias)
        if activation == "ReLU":
            return ReLU(affine)
        elif activation == "Logistic":
            return Logistic(affine)
        else:
            return affine




