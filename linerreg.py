'''
@Description  : 该类实现了线性回归基本功能
@Version      : 1.0
@Author       : gitmao2022
@Date         : 2023-03-05 22:21:44
@LastEditors  : gitmao2022
@LastEditTime : 2023-03-10 21:43:49
@FilePath     : linerreg.py
@Copyright (C) 2023 gitmao2022. All rights reserved.
'''


import numpy as np

class Linerreg:
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
        sum = np.zeros([len(self.Theta)])
        for i in range(len(self.X)):
            y_predict = np.dot(self.X[i], self.Theta)
            for j in range(len(self.Theta)):
                sum[j] += (y_predict-self.Y_true[i])*self.X[i][j]
        
        derivatives = np.array([sum[i]/len(self.X)
                               for i in range(len(self.Theta))])
        return derivatives

    def update_wb_onetime(self):
        #更新theta一次
        derivatives = self.calc_derivatives()
        self.Theta -= self.learn_rate*derivatives

    def update_wb(self):
        for i in range(self.iter_times):
            print('theta=', self.Theta)
            self.update_wb_onetime()


if __name__ == '__main__':
    X = np.random.randint(low=1, high=20, size=[20, 5])
    Theta = np.array([2, 4, 23, 7, 14], dtype=float)
    Y_true= np.dot(X,Theta)
    learn_rate = 0.0005
    iter_times = 2000
    Init_theta=np.array([0,0,0,0,0],dtype=float)
    test = Linerreg(X, Y_true, learn_rate, iter_times, Init_theta)
    test.update_wb()
