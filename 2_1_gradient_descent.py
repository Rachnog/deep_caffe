import numpy as np
import pylab as pl


class KGD(object):
    '''
        "Kid" implementation of gradient descent for two-variables functions
    '''
    def __init__(self, func, grad, x0, lmb):
        self.func = func
        self.grad = grad
        self.x0 = x0
        self.lmb = lmb

        self.history = []
        
        self.eta = 0.01


    def iteration(self, x_prev, grad):
        return x_prev - self.lmb * grad
        
        
    def iteration_momentum(self, x_prev, v_prev, grad):
         v = v_prev * (1 - self.eta) - self.lmb * grad
         return x_prev + v, v
         

    def minimize(self, num_iter):
        print 'Start stochastic gradient descent'
        self.history = []
        x = self.x0
        print x, fun(x)
        for i in xrange(num_iter):
            gradient = self.grad(x)
            x = self.iteration(x, gradient)
            self.history.append(x)
            print i, x, fun(x)


    def minimize_momentum(self, num_iter):
        print 'Start stochastic gradient descent with momentum'
        self.history = []
        x = self.x0
        v = np.array([0, 0])
        print x, fun(x)
        for i in xrange(num_iter):
            gradient = self.grad(x)
            x, v = self.iteration_momentum(x, v, gradient)
            self.history.append(x)
            print i, x, fun(x)
            if abs(0 - fun(x)) < 0.0001:
                break


    def plot(self, col):
        interval = max(abs(self.x0))
        n = 256
        x = np.linspace(-interval - 5, interval + 5, n)
        y = np.linspace(-interval - 5, interval + 5, n)
        X, Y = np.meshgrid(x, y)
        pl.contourf(X, Y, fun([X, Y]), 8, alpha=.75, cmap='jet')
        
        points = self.history        
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]

        pl.plot(xs, ys, marker='o', linestyle='--', color=str(col), label='Square')



def fun(x):
    return x[0] ** 2 + x[1] ** 2


def grad_fun(x):
    dfdx = 2 * x[0]
    dfdy = 2 * x[1]
    return np.array([dfdx, dfdy])
    
'''    
def rozenbrock(x):
    return (1-x[0])**2 + 100*(x[1] - x[0]**2)**2
    
def rozen_grad(x):
    dfdx = x[0] * (400 * x[0]**2 - 400 * x[1] + 2) - 2
    dfdy = 200 * x[1]  -200 * x[0]**2
    return np.array([dfdx, dfdy])
'''

x0 = np.array([5, 5])
lmb = 0.1

minimizer = KGD(fun, grad_fun, x0, lmb)
minimizer.minimize(10)
minimizer.plot('green')
