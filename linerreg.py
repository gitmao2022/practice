'''
@Description  : file content
@Version      : 1.0
@Author       : gitmao2022
@Date         : 2023-03-06 11:19:15
@LastEditors  : gitmao2022
@LastEditTime : 2023-03-16 17:18:06
@FilePath     : linerreg.py
@Copyright (C) 2023  by ${git_name}. All rights reserved.
'''


import numpy as np
import npas

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

        Y_predict = np.dot(self.X, self.Theta)
        Derivatives = np.dot((Y_predict-self.Y_true), self.X)/len(self.X)
        return Derivatives

    def update_wb_onetime(self):
        #更新theta一次
        derivatives = self.calc_derivatives()
        self.Theta -= learn_rate*derivatives

    def update_wb(self):
        for i in range(iter_times):
            #print('num=', i, 'theta=', self.Theta)
            self.update_wb_onetime()
        return self.Theta


if __name__ == '__main__':
    X = np.random.randint(low=1, high=20, size=[20, 5])
    X_train=X[0:10,:]
    X_test=X[11:,:]
    X_train_t=npas.linerreg_feature_scaling(X_train)
    X_test_t=npas.linerreg_feature_scaling(X_test)
    X=npas.add_right_ones(X)
    X_train_t=npas.add_right_ones(X_train_t)
    X_test_t=npas.add_right_ones(X_test_t)
    
    Theta = np.array([2, 4, 23, 7, 14, 3], dtype=float)
    Y_true = np.dot(X, Theta)
    learn_rate = 0.0005
    iter_times = 20000
    Init_theta = np.array([0, 0, 0, 0, 0, 0], dtype=float)
    test = Linerreg(X_train_t, Y_true[:10], learn_rate, iter_times, Init_theta)
    Theta_predict=test.update_wb()
    Y_test_predict=np.round(np.dot(X_test_t,Theta_predict),2)
    print("Y真实值是：",Y_true[11:])
    print("Y预测值是：",Y_test_predict)
    

