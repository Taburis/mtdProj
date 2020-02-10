
import numpy as np
from scipy import optimize
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
import time

class cftiming:
    def __init__(self):
        self.gain = 0.5
        self.delay = 1000 # delay 1000 ps
        self.method = 'tomas'

    def normalizeY(self, points):
        scale = points[np.argmax(np.abs(points))]
        return np.multiply(points, 1.0/scale)
    
    def crossFinder(self, y): 
        jmin = y.argmin()
        jmax = y.argmax()
        x = np.where(np.diff(np.sign(y)))[0]
        arr1 = np.where(x<jmax)
        if len(arr1[0]) == 0: return -2
        arr1 = arr1[0][-1]
        return int(x[arr1])
    
    def simulateCFT(self, xs, ys, dt):
        frac = self.gain 
        f = InterpolatedUnivariateSpline(xs, ys)
        def simCFT( x):
            return f(x-self.delayInterval*dt)-frac*f(x)
        return simCFT

    def getCFTiming(self, t, y):
        y = self.normalizeY(y)
        dt = t[1]-t[0]
        self.delayInterval = int(self.delay/dt) 
        ntrun =int(np.floor(2000/dt))
        ymax = np.amax(y)
        halfy = np.where(y > ymax*self.gain*0.9)[0]
        t0 = halfy[0]
        t1 = halfy[-1]+self.delayInterval
        f = self.simulateCFT(t, y, dt) 
        yy = f(t[ntrun:-ntrun])
        yys = yy[t0-ntrun:t1-ntrun]
        if len(yys) == 0 : print(len(yy), [t0-ntrun,t1-ntrun])
        x0 = self.crossFinder(yys) 
        if x0 < 0 : return -1
        inte = [t[t0+x0-1], t[t0+x0+1]]
        if self.method == 'tomas':
            try:  
                cft = optimize.toms748(f, inte[0],inte[1])
            except:
                print(inte)
                raise ValueError('failed to get CFT')
            #cft = optimize.newton(lambda x : f(x), x0=(inte[0]+inte[1])/2)
        elif self.method == 'linear':
            x1 = inte[0]
            x2 = inte[1]
            k = (f(x2)-f(x1))/(x2-x1)
            cft = x1-f(x1)/k
        return cft 
    
    def debug(self, t, y):
        y = self.normalizeY(y)
        dt = t[1]-t[0]
        ntrun =int(np.floor(2000/dt))
        ymax = np.amax(y)
        halfy = np.where(y > ymax*self.gain*0.9)[0]
        t0 = halfy[0]
        t1 = halfy[-1]+self.delayInterval
        f = self.simulateCFT(t, y, dt)
        yy = f(t[ntrun:-ntrun])
        yys = yy[t0-ntrun:t1-ntrun]
        plt.plot(t[t0-100:t1+100], f(t[t0-100:t1+100]))
        plt.plot(t[t0:t1], f(t[t0:t1]))
        plt.show()
    
    def getStrange(self, dt):
        print(np.argmax(np.abs(dt)), np.amax(np.abs(dt)))

def gaussian(x, mean, sig, amp):
    return amp*exp(-(x-mean)**2/sig**2)
