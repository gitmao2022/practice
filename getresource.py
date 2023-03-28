'''
@Description  : 获取网络各类资源
@Version      : 1.0
@Author       : gitmao2022
@Date         : 2023-03-26 23:08:09
@LastEditors  : gitmao2022
@LastEditTime : 2023-03-28 17:23:42
@FilePath     : getresource.py
@Copyright (C) 2023  by gitmao. All rights reserved.
'''
# coding: utf-8


try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import gzip
import pickle
import os
import numpy as np


dataset_dir = os.path.abspath('.')
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784



def download_with_progress(url_file_path, file_path):
    def update(count, block_size, total_size):
        percentage = int(count * block_size * 100 / total_size)
        if count % 10 == 0:
            print(f"Downloaded {count * block_size} bytes [{percentage}%]", end="\r")
    
    urllib.request.urlretrieve(url_file_path, file_path, reporthook=update)
    print(f"Downloaded {file_path} successfully!")


def download_file(url_base,file_name,local_path,rename=None,ask_when_exist=True):
    '''
    @description: 该函数用于从网络下载文件并存储于本地
    @param url_base: 文件所在的网址（不包含文件名）
    @param file_name: 文件名
    @param local_path:  本地存储路径, 不包含文件名
    @rename:如果值为none，则rename=file_name
    @ask_when_exist:当已经存在该文件时，询问是否再次下载
    @return {}
    '''

    local_file_name = local_path + "/" + rename
    if rename==None:rename=file_name

    if os.path.exists(local_file_name) and ask_when_exist==False:return
    #如果目录下已经存在要下载的文件，则选择是否覆盖
    if os.path.exists(local_file_name) and ask_when_exist==True:
        choice=input(f'当前目录下{file_name}已存在，是否重新下载？(y/n) "，是否覆盖Y/N')
        if choice.lower() == "n":
            return

    print("Downloading " + file_name + " ... ")
    download_with_progress(url_base + file_name, local_file_name)
    print("Done")

def convert_file_to_numpy(local_path,file_name,type='label',img_size=None):
    '''
    @description: 将文件转换为numpy数据
    @param ：文件夹
    @param ：文件名
    @param type：文件类型，
    @return 转换后的numpy数据
    '''

    file_path = local_path + "/" + file_name
    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        if type=='label':
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        elif type=='img':
            data = np.frombuffer(f.read(), np.uint8, offset=16)
            if img_size!=None:
                data = data.reshape(-1, img_size)
    return data

def save_data_to_pickle(data,save_path,save_name):
    with open(save_path+'/'+save_name, 'wb') as f:
        pickle.dump(data, f, -1)
    print('Create pickle file done')


def download_file_to_pickle(url_base,file_name,local_path,file_type,
                            rename=None,ask_when_exist=True):
    '''
    @description: 将网络上的文件下载下来，并变为pickle文件
    @param ：网址，不包含文件名
    @param ：文件名
    @param ：本地目录，不包含文件名
    @rename: 如果值为none，则rename=file_name
    @file_type:文件类型
    @param ask_when_exist:当已经存在pickle文件时，询问是否再次下载
    @return 
    '''
    if rename==None:rename=file_name
    download_file(url_base,file_name,local_path,rename,ask_when_exist)
    data=convert_file_to_numpy(local_path,rename,file_type)
    save_data_to_pickle(data,local_path,rename+'.pkl')



 
def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
        
    return T
    
 
def load_mnist(normalize=False, flatten=True, one_hot_label=False):
    """读入MNIST数据集
    
    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label : 
        one_hot_label为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    flatten : 是否将图像展开为一维数组
    
    Returns
    -------
    (训练图像, 训练标签), (测试图像, 测试标签)
    """
    if not os.path.exists(save_file):
        init_mnist()
        
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
            
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])
    
    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
 
    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']) 
 

'''
if __name__=='__main__':
    url_base = 'http://yann.lecun.com/exdb/mnist/'
    key_file = {
        'train_img':'train-images-idx3-ubyte.gz',
        'train_label':'train-labels-idx1-ubyte.gz',
        'test_img':'t10k-images-idx3-ubyte.gz',
        'test_label':'t10k-labels-idx1-ubyte.gz'
    }
    current_path = os.path.abspath(__file__)
    download_file_to_pickle(url_base,key_file['train_img'],
                                    current_path,file_type='img')
'''