import sys

from matplotlib import pyplot as plt
sys.path.append('../..')
import os
sys.path.append(os.getcwd())
import numpy as np
from minst.core import *

sample_size = 100
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
# batch_length = len(train_set)
batch_length=100
opt=optimizer.Optimizer(epoch=1000,sample_size=len(train_set),learning_rate=0.0002,optimizer_type='sgd')
# 构造计算图：输入向量，是一个100x1矩阵，不需要初始化，不参与训练
x =variable_node.Variable(dim=(1, 3), init=False, trainable=False)
first_layer=

# 类别标签，1男，-1女
label =variable_node.Variable(dim=(batch_length, 1), init=False, trainable=False)

# 权重向量，是一个1x3矩阵，需要初始化，参与训练
w =variable_node.Variable(dim=(3, 1), init=True, trainable=True)
# 阈值，是一个1x1矩阵，需要初始化，参与训练
b =variable_node.Variable(dim=(1, 1), init=True, trainable=True)
epoch = 1000
learning_rate = 0.0002
xw=operate_node.MatMul(x, w)
output = operate_node.Add(xw, b)
predict=activity_node.Logistic(output)
loss=loss_node.Sigmoid_Loss(predict,label)
print('label:', label.value)
accuracy=[]
for i in range(epoch):
    batch_train_set = train_set[np.random.choice(train_set.shape[0], batch_length, replace=False), :]
    x.change_dim((batch_length, 3))
    label.change_dim((batch_length, 1))
    x.set_value (batch_train_set[:, 0:3])
    label.set_value(batch_train_set[:, -1])
    loss.forward()
    w.backward(loss)
    b.backward(loss)
    w_jacobi_mean=np.mean(w.jacobi,axis=0).reshape(w.value.shape)
    b_jacobi_mean=np.mean(b.jacobi,axis=0).reshape(b.value.shape)
    w.set_value(w.value - learning_rate * w_jacobi_mean)
    b.set_value(b.value - learning_rate * b_jacobi_mean)
    w.clear_value(clear_self=False)
    b.clear_value(clear_self=False)

    x.change_dim((len(train_set), 3))
    label.change_dim((len(train_set), 1))
    x.set_value(train_set[:, 0:3])
    label.set_value(train_set[:, -1])
    predict.forward() 
    binary_predictions = (predict.value > 0.5).astype(np.int32).reshape(-1)    
    accuracy.append((train_set[:,-1] == binary_predictions).astype(np.int32).sum() / len(train_set))
    default_graph.clear_jacobi()
# 用折线图显示训练过程中的准确率
plt.plot(range(len(accuracy)), accuracy, label='accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Batch Adaline')
plt.legend()
plt.show()

