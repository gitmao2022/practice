
from .node import Node
import numpy as np 
class Variable(Node):
    """
    变量节点
    """

    def __init__(self, dim,init=True, trainable=False, **kargs):
        """
        变量节点没有父节点，构造函数接受变量的形状，是否初始化以及是否参与训练的标识
        """
        super().__init__(**kargs)
        self.dim = dim

        # 如果需要初始化，则以正态分布随机初始化变量的值
        if init:
            self.value = np.random.normal(0, 0.01, self.dim)
        # 变量节点是否参与训练
        self.trainable = trainable

    
    def change_dim(self, dim):
        """
        改变变量的形状
        """
        self.dim = dim
    
            