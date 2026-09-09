'''
@Description  : file content
@Version      : 1.1
@Author       : gitmao2022
@Date         : 2025-10-02 15:09:15
@LastEditors  : gitmao2022
@LastEditTime : 2026-05-20 17:13:55
@FilePath     : optimizer.py
@Copyright (C) 2025  by gitmao2022. All rights reserved.
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
                 learning_rate=0.001,optimizer_type='adam'):
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type.lower()
        self.train_set = train_set
        self.target_set = target_set
        self.jacobi_cache={}
        # Adam states: first/second moments and global step.
        self._adam_m = {}
        self._adam_v = {}
        self._adam_t = 0
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
                    jacobi_mean=np.mean(node.jacobi,axis=0).reshape(node.shape)   #the reason why we use reshape because the jacobi is a 2D array with shape (batch_size, node.shape) and we need to reshape it to the original shape of the node
                    self.jacobi_cache[node.node_name]=jacobi_mean
            # print(self.jacobi_cache)        
            for node in default_graph.nodes:
                if isinstance(node, Variable) and node.trainable:
                    jacobi_mean=self.jacobi_cache[node.node_name]
                    if self.optimizer_type == 'adam':
                        self._adam_t += 1
                        beta1 = 0.9
                        beta2 = 0.999
                        eps = 1e-8
                        if node.node_name not in self._adam_m:
                            self._adam_m[node.node_name] = np.zeros_like(node.value)
                            self._adam_v[node.node_name] = np.zeros_like(node.value)
                        m = self._adam_m[node.node_name]
                        v = self._adam_v[node.node_name]
                        m = beta1 * m + (1 - beta1) * jacobi_mean
                        v = beta2 * v + (1 - beta2) * (jacobi_mean ** 2)
                        m_hat = m / (1 - beta1 ** self._adam_t)
                        v_hat = v / (1 - beta2 ** self._adam_t)
                        self._adam_m[node.node_name] = m
                        self._adam_v[node.node_name] = v
                        node.set_value(node.value - self.learning_rate * m_hat / (np.sqrt(v_hat) + eps))
                    else:
                        node.set_value(node.value - self.learning_rate * jacobi_mean)

           
    def forward(self):
        """
        前向传播计算结果节点的值
        """
        # default_graph.clear_jacobi()
        # default_graph.clear_changeable_value()
        self.loss_node.forward()
      
    def add_fc_layer(self,previous_layer, back_layer_size, activation,forward_first=False):
        """
        :param previous_layer: 输入向量
        :param back_layer_size: 输出向量的维度；
        :param activation: 激活函数类型
        :param forward_first: 是否在添加层后立即进行前向传播,为后续层的输入计算提供数值支持。
        :return: 输出向量
        """
        first_layer_size = previous_layer.value.shape[1]
        weights = Variable((first_layer_size, back_layer_size), init=True, trainable=True)
        bias = Variable((1, back_layer_size), init=True, trainable=True)
        affine = Add(MatMul(previous_layer, weights), bias)
        if activation == "ReLU":
            affine=ReLU(affine)
        elif activation == "Logistic":
            affine=Logistic(affine)
        elif activation == "Softmax":
            # 由于SoftMax节点的雅可比矩阵计算存在性能问题,故在损失节点中直接计算SoftMax值并返回交叉熵损失,此处SoftMax函数仅用于计算预测值。
            p=Softmax(affine)
            affine=affine

        if forward_first:
            affine.forward()      
        return affine

    def add_conv_layer(self, previous_layer, filter_size, image_shape, activation=None, forward_first=False):
        """
        添加一个卷积层（单卷积核、same 卷积，输出尺寸与输入图像一致）。

        :param previous_layer: 输入节点，形状 (N, H*W)，N 为图像数量，每行为一张拉平的图像。
        :param filter_size: 卷积核尺寸（整数，方形核的边长）。
        :param image_shape: 二元组 (H, W)，图像真实的高和宽，用于 Convolve 内部还原形状。
        :param activation: 激活函数类型，可以是 'ReLU'、'Logistic' 等，None 表示不使用激活函数。
        :param forward_first: 是否在添加层后立即进行前向传播。
        :return: 卷积层的输出节点，形状仍为 (N, H*W)。
        """
        # 卷积核是可训练变量，形状 (filter_size, filter_size)
        kernel = Variable((filter_size, filter_size), init=True, trainable=True)
        # Convolve 节点内部完成 (N, H*W) -> (N, H, W) 卷积 -> (N, H*W) 的转换
        conv = Convolve(previous_layer, kernel, image_shape=image_shape)

        if activation == "ReLU":
            conv = ReLU(conv)
        elif activation == "Logistic":
            conv = Logistic(conv)

        if forward_first:
            conv.forward()
        return conv

    def add_pool_layer(self, previous_layer, image_shape, size=(2, 2), stride=(2, 2), forward_first=False):
        """
        添加一个最大池化层，对图像进行下采样压缩。

        :param previous_layer: 输入节点，形状 (N, H*W)。
        :param image_shape: 二元组 (H, W)，输入图像真实的高和宽。
        :param size: 二元组 (KH, KW)，池化窗口尺寸，默认 (2, 2)。
        :param stride: 二元组 (SH, SW)，步长，默认 (2, 2)。
        :param forward_first: 是否在添加层后立即进行前向传播。
        :return: 池化层输出节点，形状 (N, out_H*out_W)。
        """
        pool = MaxPooling(previous_layer, image_shape=image_shape,
                          size=size, stride=stride)
        if forward_first:
            pool.forward()
        return pool



