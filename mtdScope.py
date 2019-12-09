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

    def makeTimeAxis(self):
        self.timeAxis = np.array([nx*self.dt for nx in range(self.nsamples)], dtype=np.int64)
    
    def isEventValid(self, i):
        if i > self.nevent: 
            print('Error: Event number '+str(i)+' is invalide, the total event number: '+str(self.nevent))
            return False
        return True

    def sliceEvent(self):
        self.events = np.hsplit(self._db_, self.nevent)
        self.makeTimeAxis()

    def getEvent(self, i):
        #only return the y value for the event, need to slice the events first
        if(not self.isEventValid(i)): return []
        _data_=self.events[i]
        return _data_
    
    def getEventScaled(self, i):
        #return the scaled y value of the event
        points = self.getEvent(i)
        scaledPoints = []
        for i in range(0,self.nchannel):
            scaledPoints.append(np.multiply(points[i], self.yscale[i], dtype = np.float64))

        #time_check = time.time()
        #print("scaling consume:",time_check-time_start)
        #time_start = time_check

        return scaledPoints
    
    def getEventAdjusted(self, i):
        #time_start = time.time()
        ys = self.getEventScaled(i)
        #time_check = time.time()
        #print("reading 1 consume:",time_check-time_start)
        #time_start = time_check

        nbins = int(np.floor(1000/self.dt))
        
        points=[]
        points.append(self.timeAxis)
        for i in range(1,self.nchannel+1):
            points.append(ys[i-1])
            mean = points[i][0:nbins].mean()
            points[i] = np.subtract(points[i], mean)
        self._loadedData_ = points

        #time_check = time.time()
        #print("reading 2 consume:",time_check-time_start)
        #time_start = time_check

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
        if index == -1 : return index
        for i in range(len(y)):
            dev = abs(mean-y[i])
            if( dev > 15*std): 
                return i
        return -1

    def crossFinder(self, y):
        # return the cross index in array y
        jmin = y.argmin()
        jmax = y.argmax()
        if(jmax<jmin):
            holder=jmax
            jmax=jmin
            jmin=holder
        x = np.where(np.diff(np.sign(y)))[0]
        arr1 = np.where((x < jmax) & (x > jmin))[0]
        return int(x[arr1[0]])

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


    def signalCFT(self, x, y, ntrun, method):
        f = self.simulateCFT(x, y)
        #inte = self.crossRegion(f,n0)
        yy = f(x[ntrun:-ntrun])
        x0 = self.crossFinder(yy) 
        inte = [x[x0+ntrun], x[x0+1+ntrun]]
        if method == 'newton':
            res = optimize.newton(lambda x : f(x), x0=(inte[0]+inte[1])/2)
        elif method == 'linear':
            x1 = inte[0]
            x2 = inte[1]
            k = (f(x2)-f(x1))/(x2-x1)
            res = x1-f(x1)/k
        else: 
            raise ValueError('No method: '+method+' defined in signalCFT function')
        return res

    def getCFTiming(self, i, method):
        #time_start = time.time()
        points = self.getEventAdjusted(i)
        #time_check = time.time()
        #print("reading consume:",time_start-time_check)
        #time_start = time_check
        ntrun =int( np.floor(2000/self.dt))
        ts = []
        for channel in range(1, 3):
            res = self.signalCFT(points[0], points[channel], ntrun, method)
            ts.append(res)
        #time_check = time.time()
        #print("calculation consume:",time_start-time_check)
        #time_start = time_check
        return ts

    def getCFTiming2(self, i, method):
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

    def debug_cft(self, i, nch, method, contral):
    #def debug_cft(self, i, nch, ntrun, method):
        nrange = contral['zoomRange']
        #ntrun = contral['truncate']
        ntrun =int( np.floor(2000/self.dt))
        points = self.getEventAdjusted(i)
        f = self.simulateCFT(points[0], points[nch])
        x0 = self.crossFinder(f(points[0][ntrun:-ntrun]))
        inte = [points[0][x0+ntrun], points[0][x0+1+ntrun]]
        indexzoom = points[0][x0+ntrun-nrange:x0+ntrun+nrange]
        plt.plot(indexzoom, f(indexzoom))
        plt.axvline(inte[0],color='blue')

        if contral['check_signalCFT']: 
            ts = self.signalCFT( points[0], points[nch],ntrun, method)
            plt.axvline(ts,color='Red')

        if contral['check_getCFTiming']: 
           tsf = self.getCFTiming( i, method)
           plt.axvline(tsf[nch-1],color='Orange')
        plt.show()
        
"""
class timeWalk :
    def __init__(self):
        self.scpoe = scopeEmulator()

    def getData(self, filename ):
        self.scope.loadData(filename)
        self.scope.sliceEvent()

    def runTimeWalkAna(self, start, end, method):
        nreport = np.floor((end-start)/100)
        ts = np.zeros([self.nchannel, r1-r0], dtype = np.float32)
        timing_start = time.time()
        for i in range(r0, r1):
            if i % nreport == 0: print('processed '+str(i)+' events ('+str(np.floor((end-star)/nreport*i))+'%)...')
            res = scope.getCFTiming(i, method)
            for j in range(len(res)):
                ts[j][i-r0] = res[j]
        timing_stop = time.time()
        print('time consumed: '+str(timing_stop-timing_start))
"""
