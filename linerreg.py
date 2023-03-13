'''
@Description  : file content
@Version      : 1.0
@Author       : gitmao2022
@Date         : 2023-03-06 11:19:15
@LastEditors  : gitmao2022
@LastEditTime : 2023-03-10 10:45:12
@FilePath     : linerreg.py
@Copyright (C) 2023  by ${git_name}. All rights reserved.
'''


import numpy as np
class Linerreg:
    
    def __init__(self, X, Y_true, learn_rate, iter_times, Theta) -> None:
        self.X = X
        self.Y_true = Y_true
        self.learn_rate = learn_rate
        self.iter_times = iter_times
        self.Theta = Theta

    def calc_derivatives(self):
      
        Y_predict=np.dot(self.X,self.Theta)
        Derivatives=np.dot((Y_predict-self.Y_true),self.X)/len(self.X)
        return Derivatives

    def update_wb_onetime(self):
        #更新theta一次
        derivatives = self.calc_derivatives()
        self.Theta -= learn_rate*derivatives

    def update_wb(self):
        for i in range(iter_times):
            print('num=', i, 'theta=', self.Theta)
            self.update_wb_onetime()


if __name__ == '__main__':
    pass()
    X = np.random.randint(low=1, high=20, size=[20, 5])
    L=np.ones([len(X),1])
    X=np.column_stack((X,L))
    print(X)
    Theta = np.array([2, 4, 23, 7, 14,3], dtype=float)
    Y_true= np.dot(X,Theta)
    learn_rate = 0.0005
    iter_times =20000
    Init_theta=np.array([0,0,0,0,0,0],dtype=float)
    test = Linerreg(X, Y_true, learn_rate, iter_times, Init_theta)
    test.update_wb()
