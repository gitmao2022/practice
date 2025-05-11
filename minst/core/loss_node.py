'''
@Description  : file content
@Version      : 1.0
@Author       : gitmao2022
@Date         : 2025-03-23 16:36:36
@LastEditors  : gitmao2022
@LastEditTime : 2025-05-08 21:13:27
@FilePath     : loss_node.py
@Copyright (C) 2025  by ${git_name}. All rights reserved.
'''


import numpy as np
from .node import Node
from .activity_node import SoftMax


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
    """
    对第一个父节点施加SoftMax之后，再以第二个父节点为标签One-Hot向量计算交叉熵
    """

    def compute_value(self):
        prob = SoftMax.softmax(self.parents[0].value)
        self.value = np.mat(
            -np.sum(np.multiply(self.parents[1].value, np.log(prob + 1e-10))))

    def get_jacobi(self, parent):
        # 这里存在重复计算，但为了代码清晰简洁，舍弃进一步优化
        prob = SoftMax.softmax(self.parents[0].value)
        if parent is self.parents[0]:
            return (prob - self.parents[1].value).T
        else:
            return (-np.log(prob)).T


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
        result = np.clip(result, 1e-2, 1 - 1e-2) 
        return -label*np.log(result)-(1-label)*np.log(1-result)
    def get_jacobi(self,parent):
        label=self.parents[1].value
        result=self.parents[0].value
        result= np.clip(result, 1e-2, 1 - 1e-2)
        if parent is self.parents[0]:
            return np.diag((-label/result+(1-label)/(1-result)).flatten())
        elif parent is self.parents[1]:
            return np.diag((-np.log(result)+np.log(1-result)).flatten())
        


    