import h5py
import numpy as np
from scipy import optimize
from scipy.interpolate import interp1d
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
import time

class scopeEmulator:

    def __init__(self):
        return
        
    def loadData(self, path):
        self._f0 = h5py.File(path, 'r')
        self.attri = self._f0.attrs['Waveform Attributes'].tolist()
        self.keys = list(self._f0.keys())
        self._db_ = self._f0[self.keys[0]]
        self.nsamples = self.attri[1]
        self.dt = self.attri[3]*1.0e12 # converting to the ps unit
        self.yscale = self.attri[5]
        self.nchannel = len(self._db_)
        if(self.nchannel ==0 ): self.size = 0
        else: self.size = len(self._db_[0])
        if(self.size ==0 ): self.nevent = 0
        else: self.nevent = int(self.size/self.nsamples)
        
    def totalEvent(self): return self.nevent
    
    def isEventValid(self, i):
        if i > self.nevent: 
            print('Error: Event number '+str(i)+' is invalide, the total event number: '+str(self.nevent))
            return False
        return True
    
    def getEvent(self, i):
        if(not self.isEventValid(i)): return []
        _data_ = [np.array([nx for nx in range(i*self.nsamples, (i+1)*self.nsamples)], dtype=np.int64)]
        for data in self._db_:
            _data_.append(data[i*self.nsamples:(i+1)*self.nsamples])
        
        return _data_
    
    def getEventScaled(self, i):
        points = self.getEvent(i)
        scaledPoints = []
        scale = np.insert(self.yscale, 0, self.dt)
        for i in range(0,self.nchannel+1):
            scaledPoints.append(np.multiply(points[i], scale[i], dtype = np.float64))
        return scaledPoints
    
    def getEventAdjusted(self, i):
        points = self.getEventScaled(i)
        nbins = 20
        
        points[0] = np.subtract(points[0], points[0][0])
        for i in range(1,self.nchannel+1):
            mean = points[i][0:nbins].mean()
            points[i] = np.subtract(points[i], mean)
        self._loadedData_ = points
        return points
    
    def interpolation(self, x, y):
        return InterpolatedUnivariateSpline(x, y)
    
    def lagrangeInterpolation(self,x,y):
        return lagrange(x,y)
    
    def allType2Array(self, x):
        x = np.asarray(x)
        scalar_input= True
        if x.ndim == 0:
            x = x[None]
            scalar_input=False
        if scalar_input :
            return np.squeeze(x)
        return x
    
    def interpolationSampling(self, xs, ys):
        sampleRange = 10
        cutFreq = 1.0
        dt = self.dt
        def intpFunc(x):
            x = self.allType2Array(x)
           # if( x.any() < xs[0] or x.any() > xs[-1]):
           #     raise ValueError('Error: x value exceeded the inteploated range!')
            index = np.copy(np.floor((x-xs[0])/dt))
            
            res = np.zeros(x.shape, dtype = np.float64)
            for i in range(0, x.shape[0]):
                sampleMin = index[i]-sampleRange
                if sampleMin < 0: sampleMin = 0
                sampleMax = index[i]+sampleRange
                if sampleMax > len(xs): sampleMax = len(xs)
                for itera in range(int(sampleMin), int(sampleMax)):
                    for ifreq in range(0, int(cutFreq)):
                        arg = (index[i]- itera)*cutFreq
                        #print('arge is: ',arg)
                        res[i] += np.sinc(arg)*ys[itera]/cutFreq
                        #print(res[i])
            return res
         
        return intpFunc
    
    def simulateCFT(self,xs, ys):
        frac = 0.3
        delayInterval = int(1000/self.dt) # delay 1000 ps
        f = self.interpolation(xs, ys)
        def simCFT( x):
            return f(x-delayInterval*self.dt)-frac*f(x)
        return simCFT

    def leftJumpScanHalfSearch(self, y, mean, std):
        #do fast but not precise scan. offer the start point for leftJumpScanPrecise() 
        if(abs(y.max()) < abs(y.min())): indxright = y.argmin()
        else : indxright = y.argmax()
        indxleft = 0 
        while( indxleft+1 < indxright ):
            dev = abs(mean-y[indxleft])
            if( dev > 5*std ): return indxleft
            indxleft = int(np.floor((indxright+indxleft)/2))
        return -1

    def leftJumpScanPrecise(self, start, y, mean, std):
        index = start
        if index = -1 : return index
        for i in range(len(y)):
            dev = abs(mean-y[i])
            if( dev > 15*std): 
                return i
        return -1


    def leftJumpScanFast(self,x, y):
        nmax = 50
        stat=np.array(y[0:nmax], dtype =np.float32)
        std = stat.std()
        mean = stat.mean()
        n0 = self.leftJumpScanHalfSearch( y, mean, std)
        return self.leftJumpScanPrecise( n0, y, mean, std)

    
    def leftJumpScan(self, x, y):
        #scan the signal jump from left to right, added the argument x for debug reason
        nmax = 50
        stat=np.array(y[0:nmax], dtype =np.float32)
        std = stat.std()
        mean = stat.mean()
        index = -1
        for i in range(len(y)):
            if(i> nmax):
                dev = abs(mean-y[i])
                if( dev > 15*std): 
                    return i
        return -1
    
    def crossRegion(self, f, start):
        a0 = f(start*self.dt)
        for i in range(start, self.nsamples):
            #print(a0, f(i*self.dt))
            a1 = f(i*self.dt)
            if( a0*a1< 0 ): return [start, i]
            else :
                a0 = a1
                start = i
        return [start, -1]
    
    #def showCrossRegion(self, channel):
    #    points= self._loadedData_
    #    n0 = self.leftJumpScan(points[channel])
    #    inte = self.crossRegion(points[channel],n0)
    #    plt.plot(points[0], points[channel])
    #    print(inte[0], inte[1])
    #    plt.plot(points[0][int(inte[0]):int(inte[1])], points[channel][int(inte[0]):int(inte[1])])

    def getCFTiming(self, i, method):
        points = self.getEventAdjusted(i)
        ts = []
        for channel in range(1, 3):
            n0 = self.leftJumpScan(points[0], points[channel])
            if n0 < 0 : raise ValueError('Failed to find jump point')
            f = self.simulateCFT(points[0], points[channel])
            inte = self.crossRegion(f,n0)
            if method == 'newton':
                res = optimize.newton(lambda x : f(x), x0=self.dt*(inte[0]+inte[1])/2)
            elif method == 'linear':
                x1 = inte[0]*self.dt
                x2 = inte[1]*self.dt
                k = (f(x2)-f(x1))/(x2-x1)
                res = x1-f(x1)/k
            else: 
                raise ValueError('No method: '+method+' defined in getCFTiming function')
            ts.append(res)
        return ts

    def runTimeWalk(self, r0, r1, nstep,  method):
        # step shows report every nstep events processed
        ts = np.zeros([self.nchannel, r1-r0], dtype = np.float32)
        timing_start = time.time()
        for i in range(r0, r1):
            if i % nstep == 0: print('processing the '+str(i)+'th events...')
            res = self.getCFTiming(i, method)
            for j in range(len(res)):
                ts[j][i-r0] = res[j]
        timing_stop = time.time()
        print('time consumed: '+str(timing_stop-timing_start))
        return ts

#    def runTimeWalk(self,  method):
#        return self.runTimeWalk(self, 0, self.nevent, 100,  method)

    def showTest(self, i):
        #self.getEventAdjusted(i)
        delayInterval = 10
        #xdelay = np.subtract(xx, delayInterval*self.dt)
        f = self.interpolation(self._loadedData_[0], self._loadedData_[2])
        fs = self.simulateCFT(i)
        xx = np.copy(self._loadedData_[0][delayInterval:-delayInterval])
        #xx = np.copy(self._loadedData_[0])
        plt.plot(xx, f(xx))
        #plt.plot(xx, -0.2*f(xx))
        #plt.plot(xx, f(xdelay))
        plt.plot(xx, fs(xx))
        return
    


