'''
@Description  : file content
@Version      : 1.0
@Author       : gitmao2022
@Date         : 2025-03-23 16:36:36
@LastEditors  : gitmao2022
@LastEditTime : 2026-02-22 22:54:16
@FilePath     : loss_node.py
@Copyright (C) 2025  by ${git_name}. All rights reserved.
'''


import numpy as np
from .node import Node


class LogLoss(Node):

    def compute_value(self):

        assert len(self.parents) == 1

        x = self.parents[0].value

        return np.log(1 + np.power(np.e, np.where(-x > 1e2, 1e2, -x)))

    def get_jacobi(self, parent):
        x = parent.value
        diag = -1 / (1 + np.power(np.e, np.where(x > 1e2, 1e2, x)))

        return np.diag(diag.ravel())


class CrossEntropyWithSoftMax(Node):

    def compute_value(self):
        # 首先对parent[0]计算softmax值
        x = self.parents[0].value
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.parents[0].value = e_x / np.sum(e_x, axis=1, keepdims=True)
        v= -np.sum(np.multiply(self.parents[1].value, np.log(self.parents[0].value + 1e-10)),axis=1,keepdims=True)
        return v

    def get_jacobi(self, parent):
        #CrossEntropyWithSoftMax的父节点通常为softmax节点，其形状可能为二维。
        #因此，雅可比矩阵的计算需要考虑到这一点。
        if parent is self.parents[0]:
            #首先再次计算softmax值
            x = self.parents[0].value
            e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            prob = e_x / np.sum(e_x, axis=1, keepdims=True)
            jacobi=[]
            for row in range(parent.shape[0]):
                jacobi_row=np.zeros((self.shape[0],parent.shape[1]))
                jacobi_row[row,:]= prob[row,:] - self.parents[1].value[row,:]
                jacobi=np.concatenate((jacobi,jacobi_row),axis=1)
                
            
        elif parent is self.parents[1]:
            # 对于标签节点，雅可比矩阵相对简单，是一个对角矩阵。
            prob = self.parents[0].value
            jacobi = np.zeros((prob.shape[0], prob.shape[1], prob.shape[1]))
            for i in range(prob.shape[0]):
                p = prob[i].reshape(-1, 1)
                jacobi[i] = -np.diagflat(np.log(p + 1e-10))
            
        return jacobi
        


class PerceptionLoss(Node):
    """
    感知机损失，输入为正时为0，输入为负时为输入的相反数
    """

    def compute_value(self):
        return np.mat(np.where(
            self.parents[0].value >= 0.0, 0.0, -self.parents[0].value))

    def get_jacobi(self, parent):
        """
        雅克比矩阵为对角阵，每个对角线元素对应一个父节点元素。若父节点元素大于0，则
        相应对角线元素（偏导数）为0，否则为-1。
        """
        diag = np.where(parent.value >= 0.0, 0.0, -1)
        return np.diag(diag.ravel())

class Sigmoid_Loss(Node):
    def compute_value(self):
        label=self.parents[1].value
        result=self.parents[0].value
        #if any number in result is 0 or 1,the set the threshold to avoid log(0)
        result = np.clip(result, 1e-4, 1 - 1e-4) 
        return -label*np.log(result)-(1-label)*np.log(1-result)
    def get_jacobi(self,parent):
        label=self.parents[1].value
        result=self.parents[0].value
        result= np.clip(result, 1e-4, 1 - 1e-4)
        if parent is self.parents[0]:
            return np.diag((-label/result+(1-label)/(1-result)).flatten())
        elif parent is self.parents[1]:
            return np.diag((-np.log(result)+np.log(1-result)).flatten())
        


    