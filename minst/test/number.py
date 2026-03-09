import sys
import matplotlib
matplotlib.use('TkAgg')
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



default_graph = graph.default_graph
batch_size=30
opt=optimizer.Optimizer(epoch=200,batch_size=batch_size,train_set=train_data_list,target_set=t_train_one_hot,
                        learning_rate=0.0002,optimizer_type='sgd')
affine=opt.add_fc_layer(opt.input_var, back_layer_size=10, activation='Softmax')
opt.loss_node=loss_node.CrossEntropyWithSoftMax(affine, opt.target_var)
accuracy = []
# default_graph.draw()
for i in range(opt.epoch):
    opt.forward_backward()
    opt.forward()
    # record accuracy
    pred = np.argmax(affine.value, axis=1)
    true = np.argmax(opt.target_var.value, axis=1)
    current_loss=opt.loss_node.value.mean()
    acc = np.sum(pred == true) / batch_size
    accuracy.append(acc)
print("Training completed.","accuracy is", accuracy[-1])

#根据训练好的模型在完整测试集上评估准确率,对识别错误的样本进行可视化展示
opt.input_var.set_value(test_data_list)  # set test data as input
opt.target_var.set_value(t_test_one_hot)  # set test labels as target
opt.forward()  # forward pass to compute predictions
test_pred = np.argmax(affine.value, axis=1)
test_true = np.argmax(opt.target_var.value, axis=1)
test_acc = np.sum(test_pred == test_true) / test_data_list.shape[0]
print("Test accuracy:", test_acc)
# 可视化错误样本
error_indices = np.where(test_pred != test_true)[0]
if len(error_indices) > 0:
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(error_indices[:25]):  # show up to 25 errors
        plt.subplot(5, 5, i + 1)
        plt.imshow(test_data_list[idx].reshape(28, 28), cmap='gray')
        plt.title(f"True: {test_true[idx]}, Pred: {test_pred[idx]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()




# # GUI for prediction
# import tkinter as tk
# from tkinter import messagebox

# class MNIST_GUI:
#     def __init__(self, master):
#         self.master = master
#         master.title("MNIST Predictor")
        
#         # 28x28 grid, scale factor 10 -> 280x280
#         self.scale = 10
#         self.width = 28 * self.scale
#         self.height = 28 * self.scale
        
#         self.canvas = tk.Canvas(master, width=self.width, height=self.height, bg='black')
#         self.canvas.pack()
#         self.canvas.bind("<B1-Motion>", self.paint)
#         self.canvas.bind("<Button-1>", self.paint)
        
#         self.grid_data = np.zeros((28, 28))
        
#         btn_frame = tk.Frame(master)
#         btn_frame.pack(pady=10)
        
#         self.predict_btn = tk.Button(btn_frame, text="Predict", command=self.predict, width=10)
#         self.predict_btn.pack(side=tk.LEFT, padx=5)
        
#         self.clear_btn = tk.Button(btn_frame, text="Clear", command=self.clear, width=10)
#         self.clear_btn.pack(side=tk.LEFT, padx=5)
        
#         self.label = tk.Label(master, text="Draw a digit", font=("Helvetica", 16))
#         self.label.pack(pady=5)

#     def paint(self, event):
#         col = event.x // self.scale
#         row = event.y // self.scale
        
#         if 0 <= col < 28 and 0 <= row < 28:
#             # Draw on canvas
#             x1, y1 = col * self.scale, row * self.scale
#             x2, y2 = (col + 1) * self.scale, (row + 1) * self.scale
#             self.canvas.create_rectangle(x1, y1, x2, y2, fill='white', outline='white')
            
#             # Update grid data (normalized 0-255 like training data)
#             self.grid_data[row, col] = 255.0
            
#             # Simple brush size (3x3) to make lines thicker
#             for r in range(max(0, row-1), min(28, row+2)):
#                 for c in range(max(0, col-1), min(28, col+2)):
#                      self.grid_data[r, c] = 255.0
#                      rx1, ry1 = c * self.scale, r * self.scale
#                      rx2, ry2 = (c + 1) * self.scale, (r + 1) * self.scale
#                      self.canvas.create_rectangle(rx1, ry1, rx2, ry2, fill='white', outline='white')


#     def clear(self):
#         self.canvas.delete("all")
#         self.grid_data.fill(0)
#         self.label.config(text="Draw a digit")

#     def predict(self):
#         try:
#             # Prepare input: flatten to (1, 784)
#             input_vec = self.grid_data.reshape(1, 784)
            
#             # Create a NEW variable node for single prediction to avoid shape conflicts?
#             # Or just set value. Variable logic handles shape changes if nodes support it.
#             # MatMul (Add's parent) supports variable batch dimension.
#             # Add supports broadcasting bias.
#             # So updating `opt.input_var` should work.
            
#             opt.input_var.set_value(input_vec)
            
#             # Forward pass from affine node to get logits
#             affine.forward()
            
#             # Get prediction
#             logits = affine.value
#             prediction = np.argmax(logits, axis=1)[0]
            
#             self.label.config(text=f"Prediction: {prediction}")
            
#         except Exception as e:
#             messagebox.showerror("Error", str(e))

# root = tk.Tk()
# app = MNIST_GUI(root)
# root.mainloop()




