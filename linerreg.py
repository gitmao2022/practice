import numpy as np

class Linerreg:
    def __init__(self, x, y_true, learn_rate, iter_times, w, b) -> None:
        self.x = x
        self.y_true = y_true
        self.learn_rate = learn_rate
        self.iter_times = iter_times
        self.w = w
        self.b = b

    def calc_derivatives(self):
        # calc derivatives
        sum_w, sum_b = 0, 0
        nums = len(self.x)
        for i in range(nums):
            sum_w += (self.y_true[i]-self.w*self.x[i]-self.b)*self.x[i]
            sum_b += self.y_true[i]-self.w*self.x[i]-self.b
        derivatives_w = sum_w/nums
        derivatives_b = sum_b/nums
        return derivatives_w, derivatives_b

    def update_wb_onetime(self):
        #更新w、b一次
        (derivatives_w, derivatives_b) = self.calc_derivatives()
        self.w -= self.learn_rate*derivatives_w
        self.b -= self.learn_rate*derivatives_b

    def update_wb(self):
        for i in range(self.iter_times):
            print('w=', self.w, 'b=', self.b)
            self.update_wb_onetime()


if __name__ =='__main__':
    x=np.random.randint(low=1,high=100,size=[20])
    bias=np.random.random([len(x)])
    y_true=5*x+24-bias
    learn_rate=0.01
    iter_times=500
    w,b=0,0
    test=Linerreg(x, y_true, learn_rate, iter_times, w, b)
    test.update_wb()
