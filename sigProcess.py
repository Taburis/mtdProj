
import numpy as np
from scipy import optimize
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
import time

def normalizeY(points):
    scale = points[np.argmax(np.abs(points))]
    return np.multiply(points, 1.0/scale)

def crossFinder(y): 
        jmin = y.argmin()
        jmax = y.argmax()
        x = np.where(np.diff(np.sign(y)))[0]
        arr1 = np.where(x<jmax)
        if len(arr1[0]) == 0: return -1
        arr1 = arr1[0][-1]
        return int(x[arr1])

def simulateCFT(xs, ys, dt):
        frac = 0.5
        delayInterval = int(1000/dt) # delay 1000 ps
        f = InterpolatedUnivariateSpline(xs, ys)
        def simCFT( x):
            return f(x-delayInterval*dt)-frac*f(x)
        return simCFT

def getCFTiming(t, y):
        y = normalizeY(y)
        dt = t[1]-t[0]
        ntrun =int(np.floor(2000/dt))
        f = simulateCFT(t, y, dt)
        yy = f(t[ntrun:-ntrun])
        x0 = crossFinder(yy) 
        if x0 < 0 : return -1
        inte = [t[x0+ntrun], t[x0+1+ntrun]]
        x1 = inte[0]
        x2 = inte[1]
        k = (f(x2)-f(x1))/(x2-x1)
        cft = x1-f(x1)/k
        return cft


