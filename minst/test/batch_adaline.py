import sys
sys.path.append('../..')
import os
sys.path.append(os.getcwd())
import numpy as np
from minst.core import *

male_heights = np.random.normal(171, 6, 500)
female_heights = np.random.normal(158, 5, 500)

male_weights = np.random.normal(70, 10, 500)
female_weights = np.random.normal(57, 8, 500)

male_bfrs = np.random.normal(16, 2, 500)
female_bfrs = np.random.normal(22, 2, 500)

male_labels = [1] * 500
female_labels = [0] * 500

train_set = np.array([np.concatenate((male_heights, female_heights)),
                      np.concatenate((male_weights, female_weights)),
                      np.concatenate((male_bfrs, female_bfrs)),
                      np.concatenate((male_labels, female_labels))]).T
np.random.shuffle(train_set)

batch_length = len(train_set)
# 构造计算图：输入向量，是一个100x1矩阵，不需要初始化，不参与训练
x =variable_node.Variable(dim=(batch_length, 3), init=False, trainable=False)
x.set_value (train_set[:, 0:3])
# 类别标签，1男，-1女
label =variable_node.Variable(dim=(batch_length, 1), init=False, trainable=False)
label.set_value(train_set[:, -1])
# print(label.value)
# print(x.value)

# 权重向量，是一个1x3矩阵，需要初始化，参与训练
w =variable_node.Variable(dim=(3, 1), init=True, trainable=True)

# 阈值，是一个1x1矩阵，需要初始化，参与训练
b =variable_node.Variable(dim=(1, 1), init=True, trainable=True)


xw=operate_node.MatMul(x, w)
output = operate_node.Add(xw, b)
predict=activity_node.Logistic(output)
loss=loss_node.Sigmoid_Loss(predict,label)
loss.forward()
# print('label shape:',label.value.shape,'predict shape:',predict.value.shape)
print('loss.value',loss.value)

