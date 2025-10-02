'''
@Description  : file content
@Version      : 1.0
@Author       : gitmao2022
@Date         : 2025-10-02 15:09:15
@LastEditors  : gitmao2022
@LastEditTime : 2025-10-02 16:06:49
@FilePath     : optimizer.py
@Copyright (C) 2025  by ${gitmao2022}. All rights reserved.
'''

from .graph import *
from .node import *
from .operate_node import *
from .loss_node import *
from .activity_node import *
from .variable_node import *

class Optimizer:
    def __init__(self, epoch=1000,learning_rate=0.01,optimizer_type='adam',):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
    
    def add_fc_layer(self,input, input_size, size, activation):
        """
        :param input: 输入向量
        :param input_size: 输入向量的维度
        :param size: 神经元个数，即输出个数（输出向量的维度）
        :param activation: 激活函数类型
        :return: 输出向量
        """
        weights = Variable((size, input_size), init=True, trainable=True)
        bias = Variable((size, 1), init=True, trainable=True)
        affine = Add(MatMul(weights, input), bias)
        if activation == "ReLU":
            return ReLU(affine)
        elif activation == "Logistic":
            return Logistic(affine)
        else:
            return affine




