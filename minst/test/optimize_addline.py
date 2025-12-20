import sys

from matplotlib import pyplot as plt
sys.path.append('../..')
import os
sys.path.append(os.getcwd())
import numpy as np
from minst.core import *


sample_size = 100
male_heights = np.random.normal(171, 6, sample_size)
female_heights = np.random.normal(158, 5, sample_size)

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
target_set=train_set[:,-1]
train_set = train_set[:,:-1]

default_graph = graph.default_graph
opt=optimizer.Optimizer(epoch=2000,batch_size=len(train_set),train_set=train_set,target_set=target_set,
                        learning_rate=0.0001,optimizer_type='sgd')
affine=opt.add_fc_layer(opt.input_var, back_layer_size=1, activation='Logistic')
opt.loss_node=loss_node.Sigmoid_Loss(affine, opt.target_var)
# 记录训练过程中 weights、bias 以及 accuracy
weights_history = []
bias_history = []
accuracy = []
# default_graph.draw()
for i in range(opt.epoch):
    opt.forward_backward()
    # default_graph.draw()
    opt.forward()

    # 从计算图中取得权重和偏置变量：affine = Add(MatMul(previous, weights), bias)
    matmul_node = affine.parents[0].parents[0]
    weight_var = matmul_node.parents[1]
    bias_var = affine.parents[0].parents[1]

    # 保存当前权重（展平为向量）和偏置（标量或向量）
    weights_history.append(np.array(weight_var.value).flatten().copy())
    bias_arr = np.array(bias_var.value).flatten()
    # 如果 bias 是向量，取第一个元素；保持为标量序列
    bias_history.append(float(bias_arr[0]) if bias_arr.size > 0 else 0.0)

    # 计算当前准确率并记录
    binary_predictions = (affine.value.reshape(-1) > 0.5)
    accuracy.append((target_set== binary_predictions).astype(np.int32).sum() / len(target_set))
    default_graph.clear_jacobi()

# 把记录转为 numpy 数组，方便绘图
weights_history = np.array(weights_history)
bias_history = np.array(bias_history)
accuracy = np.array(accuracy)

# 绘制：权重的每个分量和 bias 在左 y 轴，accuracy 在右 y 轴
epochs = np.arange(len(accuracy))
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制每个权重分量
if weights_history.ndim == 2:
    for j in range(weights_history.shape[1]):
        ax1.plot(epochs, weights_history[:, j], label=f'weight_{j}')
else:
    ax1.plot(epochs, weights_history, label='weight')

# 绘制 bias
ax1.plot(epochs, bias_history, label='bias', linestyle='--')
ax1.set_xlabel('epoch')
ax1.set_ylabel('weights / bias')

# accuracy 使用次坐标轴
ax2 = ax1.twinx()
ax2.plot(epochs, accuracy, color='k', label='accuracy')
ax2.set_ylabel('accuracy')
ax2.set_ylim(0.0, 1.0)

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title('Training curves: weights, bias and accuracy')
plt.tight_layout()
try:
    # 确保在多数环境下窗口保持打开（阻塞直到关闭）
    plt.show(block=True)
except TypeError:
    # 老版本 matplotlib 可能不支持 block 参数，退回到非阻塞并等待用户确认
    plt.show()
    input("按回车键退出并关闭窗口...")


