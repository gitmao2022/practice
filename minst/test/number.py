import sys
from matplotlib import pyplot as plt
# add project root (parent of `nemat`) so `import nemat` works
sys.path.append('../..')
import os
import numpy as np
from minst.core import *
# replace slow readers with fast, cached readers
def read_images_fast(filename, max_items=None):
    npy_file = filename + '.npy'
    if os.path.exists(npy_file):
        data = np.load(npy_file)
    else:
        with open(filename, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            num = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
        data = np.fromfile(filename, dtype=np.uint8, offset=16)
        data = data.reshape(num, rows * cols)
        np.save(npy_file, data)
    if max_items is not None:
        return data[:max_items]
    return data

def read_labels_fast(filename, max_items=None):
    npy_file = filename + '.npy'
    if os.path.exists(npy_file):
        labels = np.load(npy_file)
    else:
        with open(filename, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            num = int.from_bytes(f.read(4), 'big')
        labels = np.fromfile(filename, dtype=np.uint8, offset=8)
        np.save(npy_file, labels)
    if max_items is not None:
        return labels[:max_items]
    return labels

# 从本地目录中文件train-images-idx3-ubyte以及train-labels-idx1-ubyte加载手写数字数据集
train_data_list = read_images_fast('./train-images-idx3-ubyte', max_items=1000)
test_data_list = read_images_fast('./t10k-images-idx3-ubyte', max_items=1000)
train_label_list = read_labels_fast('./train-labels-idx1-ubyte', max_items=1000)
test_label_list = read_labels_fast('./t10k-labels-idx1-ubyte', max_items=1000)

# transform t_train and  t_test to one-hot vectors
t_train_one_hot = np.zeros((train_label_list.shape[0], 10))
for i in range(train_label_list.shape[0]):
    t_train_one_hot[i][train_label_list[i]] = 1
t_test_one_hot = np.zeros((test_label_list.shape[0], 10))
for i in range(test_label_list.shape[0]):
    t_test_one_hot[i][test_label_list[i]] = 1    
# get dimension of train_data_list
# for row in x_train[1]:
#     print(' '.join(map(str, row)), end='\n')


default_graph = graph.default_graph
batch_size=30
opt=optimizer.Optimizer(epoch=1000,batch_size=batch_size,train_set=train_data_list,target_set=t_train_one_hot,
                        learning_rate=0.0002,optimizer_type='sgd')
affine=opt.add_fc_layer(opt.input_var, back_layer_size=10, activation='SoftMax')
opt.loss_node=loss_node.CrossEntropyWithSoftMax(affine, opt.target_var)
accuracy = []
# default_graph.draw()
for i in range(opt.epoch):
    opt.forward_backward()
    opt.forward()
    # record accuracy
    pred = np.argmax(affine.value, axis=1)
    true = np.argmax(opt.target_var.value, axis=1)
    acc = np.sum(pred == true) / batch_size
    accuracy.append(acc)
    print(f'Epoch {i}, Accuracy: {acc}')




