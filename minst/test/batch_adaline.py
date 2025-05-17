import sys

from matplotlib import pyplot as plt
sys.path.append('../..')
import os
sys.path.append(os.getcwd())
import numpy as np
from minst.core import *

sample_size = 50
male_heights = np.random.normal(171, 6, sample_size)
female_heights = np.random.normal(158, 50, sample_size)

male_weights = np.random.normal(70, 10, sample_size)
female_weights = np.random.normal(57, 8, sample_size)

male_bfrs = np.random.normal(16, 2, sample_size)
female_bfrs = np.random.normal(22, 2, sample_size)

male_labels = [1] * sample_size
female_labels = [0] * sample_size

train_set = np.array([np.concatenate((male_heights, female_heights)),
                      np.concatenate((male_weights, female_weights)),
                      np.concatenate((male_bfrs, female_bfrs)),
                      np.concatenate((male_labels, female_labels))]).T
np.random.shuffle(train_set)

default_graph = graph.default_graph
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

learning_rate = 0.0002
xw=operate_node.MatMul(x, w)
output = operate_node.Add(xw, b)
predict=activity_node.Logistic(output)
loss=loss_node.Sigmoid_Loss(predict,label)
print('label:', label.value)
accuracy=[]
for i in range(300):
    # print('w:', w.value)
    # print('b:', b.value)
    loss.forward()
    # print('output:', output.value)
    # print('predict:', predict.value)
    # print('epoch:', i, 'loss:', np.sum(loss.value))
    # print('activation:', predict.value.T)
    w.backward(loss)
    b.backward(loss)
    # default_graph.draw()
    w_temp_value=w.value.copy()
    b_temp_value=b.value.copy()
    w.clear_value()
    b.clear_value()
    w_jacobi_mean=np.mean(w.jacobi,axis=0).reshape(w_temp_value.shape)
    b_jacobi_mean=np.mean(b.jacobi,axis=0).reshape(b_temp_value.shape)
    # print("w_jacobi_mean",w_jacobi_mean)
    # print("b_jacobi_mean",b_jacobi_mean)
    w.set_value(w_temp_value - learning_rate * w_jacobi_mean)
    b.set_value(b_temp_value - learning_rate * b_jacobi_mean)
    predict.forward() 
    binary_predictions = (predict.value > 0.5).astype(np.int32).reshape(-1)    
    accuracy.append((train_set[:,-1] == binary_predictions).astype(np.int32).sum() / len(train_set))
    # print('accuracy=',accuracy)
    default_graph.clear_jacobi()
# 用折线图显示训练过程中的准确率
plt.plot(range(len(accuracy)), accuracy, label='accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Batch Adaline')
plt.legend()
plt.show()

