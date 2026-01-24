import sys
from matplotlib import pyplot as plt
# add project root (parent of `nemat`) so `import nemat` works
sys.path.append('../..')
import os
import numpy as np
from minst.core import *
#从本地目录中文件train-images-idx3-ubyte以及train-labels-idx1-ubyte加载手写数字数据集
def read_pic_from_file(filename):
    f = open(filename, 'rb')
    magic = int.from_bytes(f.read(4), 'big')
    num = int.from_bytes(f.read(4), 'big')
    width = int.from_bytes(f.read(4), 'big')
    height = int.from_bytes(f.read(4), 'big')

    dataset = []
    for i in range(num):
        pic = []
        for i in range(width * height):
            pic.append(int.from_bytes(f.read(1), 'big'))
        dataset.append(pic)
        # show_pic(pic)
    return dataset

train_data_list = read_pic_from_file('./train-images-idx3-ubyte')
test_data_list = read_pic_from_file('./t10k-images-idx3-ubyte')

def read_label_from_file(filename):
    f = open(filename, 'rb')
    magic = int.from_bytes(f.read(4), 'big')
    num = int.from_bytes(f.read(4), 'big')
    
    labels = []
    for i in range(num):
        labels.append(int.from_bytes(f.read(1), 'big'))
    return labels
train_label_list = read_label_from_file('./train-labels-idx1-ubyte')
test_label_list = read_label_from_file('./t10k-labels-idx1-ubyte')


#只取前1000个样本
x_train = np.array(train_data_list[0:1000])
t_train = np.array(train_label_list[0:1000])
x_test = np.array(test_data_list[0:1000])
t_test= np.array(test_label_list[0:1000])
#transform t_train and  t_test to one-hot vectors
t_train_one_hot = np.zeros((t_train.shape[0], 10))
for i in range(t_train.shape[0]):
    t_train_one_hot[i][t_train[i]] = 1
t_test_one_hot = np.zeros((t_test.shape[0], 10))
for i in range(t_test.shape[0]):
    t_test_one_hot[i][t_test[i]] = 1    
#get dimension of train_data_list
# for row in x_train[1]:
#     print(' '.join(map(str, row)), end='\n')


default_graph = graph.default_graph
batch_size=30
opt=optimizer.Optimizer(epoch=1000,batch_size=batch_size,train_set=x_train,target_set=t_train_one_hot,
                        learning_rate=0.0002,optimizer_type='sgd')
affine=opt.add_fc_layer(opt.input_var, back_layer_size=10, activation='SoftMax')
opt.loss_node=loss_node.CrossEntropyWithSoftMax(affine, opt.target_var)
accuracy = []
# default_graph.draw()
for i in range(opt.epoch):
    opt.forward_backward()
    opt.forward()
    #record accuracy
    pred = np.argmax(affine.output, axis=1)
    true = np.argmax(opt.target_var.data, axis=1)
    acc = np.sum(pred == true) / batch_size
    accuracy.append(acc)
    print(f'Epoch {i}, Loss: {opt.loss_node.output}, Accuracy: {acc}')

    


