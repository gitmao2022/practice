'''
@Description  : file content
@Version      : 1.0
@Author       : gitmao2022
@Date         : 2023-03-25 16:59:51
@LastEditors  : gitmao2022
@LastEditTime : 2023-06-19 17:57:09
@FilePath     : logisticreg.py
@Copyright (C) 2023  by ${gitmao}. All rights reserved.
'''

import numpy as np
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

import npas



class logisticreg:

    def __init__(self, X, Y_true, learn_rate, iter_times, Theta) -> None:
        '''
        @param learn_rate {float}: 学习率
        @param iter_times {int}: 训练次数
        '''
        self.X = X
        self.Y_true = Y_true
        self.learn_rate = learn_rate
        self.iter_times = iter_times
        self.Theta = Theta


    def calc_derivatives(self):

        Y_predict = npas.sigmoid(np.dot(self.X, self.Theta))
        Derivatives = np.dot((Y_predict - self.Y_true), self.X) / len(self.X)
        return Derivatives

    def update_wb_onetime(self):
        #更新theta一次
        Derivatives = self.calc_derivatives()
        self.Theta -= self.learn_rate * Derivatives

    def update_wb(self):
        for i in range(self.iter_times):
            #print('num=', i, 'theta=', self.Theta)
            self.update_wb_onetime()
        return self.Theta
    
    