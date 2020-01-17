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
#    def loadData(self, path):
#        self._f0 = h5py.File(path, 'r')
#        self.attri = self._f0.attrs['waveform'].tolist()
#        #self.attri = self._f0.attrs['Waveform Attributes'].tolist()
#        self.keys = list(self._f0.keys())
#        self._db_ = self._f0[self.keys[0]]
#        self.nsamples = self.attri[1]
#        self.dt = self.attri[3]*1.0e12 # converting to the ps unit
#        self.yscale = self.attri[5]
#        self.nchannel = len(self._db_)
#        if(self.nchannel ==0 ): self.size = 0
#        else: self.size = len(self._db_[0])
#        if(self.size ==0 ): self.nevent = 0
#        else: self.nevent = int(self.size/self.nsamples)

    def loadData(self, path):
        self._f0=h5py.File(path, 'r')
        self._db_ = self._f0['waveform']
        self.attrs = self._db_.attrs
        self.nsamples = self.attrs['nPt']
        self.dt = self.attrs['dt']*1.0e12
        self.nevent = int(np.array(self._db_).shape[1]/self.nsamples)
        self.scale_channels = []
        self.yscale = []
        self.zeros = []
        self.bit_size = []
        self.nchannel=0 
        for i in range(4):
            if not self.attrs['chmask'][i] : continue
            self.scale_channels.append(self.attrs['vertical'+str(i+1)])
            self.nchannel=self.nchannel+1
        for i in range(self.nchannel):
            self.yscale.append(self.scale_channels[i][0])
            self.zeros.append(self.scale_channels[i][1])
            self.bit_size.append(self.scale_channels[i][2])

        
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
            scaledPoints.append(np.multiply(np.subtract(points[i], self.zeros[i]),self.yscale[i], dtype = np.float64))

        #time_check = time.time()
        #print("scaling consume:",time_check-time_start)
        #time_start = time_check

        return scaledPoints
    
    def getEventAdjusted(self, i):
        ys = self.getEventScaled(i)
#        #time_check = time.time()
#        #print("reading 1 consume:",time_check-time_start)
#        #time_start = time_check
#
#        nbins = int(np.floor(1000/self.dt))
#        
        points=[]
        points.append(self.timeAxis)
        for i in range(self.nchannel):
            points.append(ys[i])

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

    def runTimeWalk(self, r0, r1, nstep,  method, error_tolerance = 500):
        # step shows report every nstep events processed
        ts = np.zeros([self.nchannel, r1-r0], dtype = np.float32)
        timing_start = time.time()
        position = 0
        for i in range(r0, r1):
            if i % nstep == 0: print('processing the '+str(i)+'th events...')
            res = self.getCFTiming(i, method)
#            if position !=0 and abs(abs(res[1]-res[0])-position) > error_tolerance: 
#                print("error occurs at:",i)
#                break
#            else: position = abs(res[1]-res[0])
            for j in range(len(res)):
                ts[j][i-r0] = res[j]
        timing_stop = time.time()
        print('time consumed: '+str(timing_stop-timing_start))
        return ts

    def runTimeWalk_peak2peak_filtered(self, r0, r1, nstep,  method, peak, pkrage):
        ts = np.zeros([self.nchannel, r1-r0], dtype = np.float32)
        timing_start = time.time()
        for i in range(r0, r1):
            if i % nstep == 0: print('processing the '+str(i)+'th events...')
            points = self.getEventAdjusted(i)
            if self.peak2peak_cut(peak, pkrage, points[2]): continue
            res = self.getCFTiming(i, method)
            for j in range(len(res)):
                ts[j][i-r0] = res[j]
        timing_stop = time.time()
        print('time consumed: '+str(timing_stop-timing_start))
        return ts

    def debug_cft(self, i, nch, method, contral):
    #def 50debug_cft(self, i, nch, ntrun, method):
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
        
    def peak2peak_cut(self, cuts, ranges, points):
        #event = np.array(points, dtype =np.float32)
        maxvalue = np.amax(points)
        minvalue = np.amin(points)
        cc = abs(maxvalue - minvalue )
        if cc < cuts-ranges: return True
        elif cc > cuts+ranges: return True
        return False

    def integration_over_events(self, r0, r1, channel):
        integ = []
        for i in range(r0, r1):
            points = self.getEventAdjusted(i)
            t = points[0]
            signal = points[channel]
            integ.append(np.trapz(signal,x = t))
        return integ
    
    def showEvent(self, i, channels):
        points = self.getEventAdjusted(i)
        p0 = np.zeros((self.nsamples,1))
        for i in channels:
            plt.plot(points[0], points[i], label='channel: '+str(i))
        plt.plot(points[0], p0, '--')
        plt.xlabel('time (ps)')
        plt.legend(loc='upper right')
        plt.show()

    def trigger_check(self, points, trig, channels):
        #return 1 for skip
        for i in channels:
            if np.amax(np.absolute(points[i]))< trig: return True
        return False

    def charge_convertion(self, r0, r1, chan, trig, channels, do_correction):
        evt = []
        integ = [] 
        cnn = channels
        cnn.append(chan)
        for i in range(r0, r1):
            points = self.getEventAdjusted(i)
            if do_correction:
                self.cnn_baseline_correction(points, cnn)
            if self.trigger_check(points, trig, channels) : continue
            t = points[0]
            signal = points[chan]
            #integ[str(i)] = np.trapz(signal,x = t)
            evt.append(i)
            integ.append(np.trapz(signal,x = t))
        return evt, integ
            
    def baseline(self, points):
        ps = np.absolute(points)
        

    def load_cnn_baseline_finder(self,path):
        import tensorflow as tf
        model = tf.keras.models.load_model(path)
        self.baseline_finder = model
        self.baseline_finder.summary()

    def normalized_input(self,points):
        raw = points
        amin = np.amin(raw)
        amax = np.amax(raw)
        sign = 1
        if abs(amax) < abs(amin):
            sign = -1
            raw = np.negative(raw)
        size0 = raw.shape[0]
        size = 1000
        npts = int(size0/size)
        data = np.zeros(size)
        start = 0
        for i in range(size):
            count = 0
            for j in range(start, size0):
                if count == npts: 
                    start = j
                    break
                else :
                    data[i]+=raw[j]
                    count+=1
        amax = np.amax(data)
        data = np.multiply(data,1./amax)
        return sign*amax/npts, data

    def cnn_baseline_correction(self,points, channels):
        for i in channels:
            input_data = points[i]
            shape = input_data.shape
            scale, data = self.normalized_input(input_data)
            base = self.baseline_finder.predict(data.reshape(1, 1000,1))
            base = scale*base
            points[i] = np.subtract(points[i], base).reshape(shape[0])

