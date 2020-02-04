import h5py
import numpy as np
from scipy import optimize
from scipy.interpolate import interp1d
from scipy import integrate
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
import time
from tqdm import tqdm

class scopeEmulator:

    def __init__(self):
        return

    def loadData(self, path):
        self.file_name=path.split('/')[-1]
        self._f0=h5py.File(path, 'r')
        self._db_ = self._f0['waveform']
        self.attrs = self._db_.attrs
        self.nsamples = self.attrs['nPt']
        self.dt = self.attrs['dt']*1.0e12
        self.nevent = int(np.array(self._db_).shape[1]/self.nsamples)
        self.scale_channels = []
        self.yscale = []
        self.zeros = []
        self.badEvent = False
        self.bit_size = []
        self.nchannel=0 
        self.trigger = self.no_trigger
        for i in range(4):
            if not self.attrs['chmask'][i] : continue
            self.scale_channels.append(self.attrs['vertical'+str(i+1)])
            self.nchannel=self.nchannel+1
        for i in range(self.nchannel):
            self.yscale.append(self.scale_channels[i][0])
            self.zeros.append(self.scale_channels[i][1])
            self.bit_size.append(self.scale_channels[i][2])

    def totalEvent(self): return self.nevent

    def no_trigger(self, points): return 1 # always trigger

    def makeTimeAxis(self):
        self.t = np.array([nx*self.dt for nx in range(self.nsamples)], dtype=np.int64)
    
    def isEventValid(self, i):
        if i > self.nevent: 
            print('Error: Event number '+str(i)+' is invalide, the total event number: '+str(self.nevent))
            return False
        return True

    def sliceEvent(self):
        self.events = np.hsplit(self._db_, self.nevent)
        self.makeTimeAxis()

    def getEvent(self, i):
        self.badEvent = False
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
  
    def triggeredEvent(self, i):
        points = self.getEventAdjusted(i)
        if self.trigger(points): return points, True
        else : return points, False
   
    def getEventAdjusted(self, i):
        ys = self.getEventScaled(i)
#        #time_check = time.time()
#        #print("reading 1 consume:",time_check-time_start)
#        #time_start = time_check
#
#        nbins = int(np.floor(1000/self.dt))
#        
        points=[]
        points.append(self.t)
        for i in range(self.nchannel):
            points.append(ys[i])

        #time_check = time.time()
        #print("reading 2 consume:",time_check-time_start)
        #time_start = time_check

        return points

    def normalizeY(self, points):
        scale = points[np.argmax(np.abs(points))]
        return np.multiply(points, 1.0/scale)

    def getEventYNormalized(self, i):
        points = self.getEventAdjusted(i)
        for i in range(self.nchannel):
            points[i+1] = self.normalizeY(points[i+1])
        return points
    
    def interpolation(self, x, y):
        return InterpolatedUnivariateSpline(x, y)
    
    def simulateCFT(self,xs, ys):
        frac = 0.5
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
        x = np.where(np.diff(np.sign(y)))[0]
        arr1 = np.where(x<jmax)
        if len(arr1[0]) == 0 : 
            self.badEvent = True
            return 0
        arr1 = arr1[0][-1]
        return int(x[arr1])

    def signalCFT(self, x, y, ntrun, method):
        f = self.simulateCFT(x, y)
        yy = f(x[ntrun:-ntrun])
        x0 = self.crossFinder(yy) 
        if self.badEvent : return 0
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
        points = self.getEventYNormalized(i)
        ntrun =int(np.floor(2000/self.dt))
        ts = []
        for channel in range(1, self.nchannel+1):
            res = self.signalCFT(points[0], points[channel], ntrun, method)
            if self.badEvent : 
                return []
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
            if self.badEvent: 
                print("collapsed at event:",i,"skipped")
                continue
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

    def debug_cft(self, i, nch, method, nrange):
    #def 50debug_cft(self, i, nch, ntrun, method):
        #ntrun = contral['truncate']
        ntrun =int( np.floor(2000/self.dt))
        points = self.getEventYNormalized(i)
        f = self.simulateCFT(points[0], points[nch])
        x0 = self.crossFinder(f(points[0][ntrun:-ntrun]))
        inte = [points[0][x0+ntrun], points[0][x0+1+ntrun]]
        indexzoom = points[0][x0+ntrun-nrange:x0+ntrun+nrange]
        plt.plot(indexzoom, f(indexzoom))
        plt.axvline(inte[0],color='blue')

        ts = self.signalCFT( points[0], points[nch],ntrun, method)
        plt.axvline(ts,color='Red')

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

    def charge_convertion(self, r0, r1, chan,trigger, do_correction):
        evt = []
        integ = [] 
        t = self.t
        for i in range(r0, r1):
            points = self.getEventAdjusted(i)
            if do_correction:
                self.cnn_baseline_correction(points)
            if trigger(points) : continue
            signal = points[chan]
            #print("len(t): ",len(t))
            #print("shape signal: ",signal.shape)
            #integ[str(i)] = np.trapz(signal,x = t)
            evt.append(i)
            integ.append(np.trapz(signal,x = t))
        return evt, integ
            

    def load_cnn_baseline_finder(self,path):
        import baseline_cnn_model as bf
        self.baseline_model = bf.baseline_finder(path)

    def cnn_baseline_correction(self,points):
        for i in range(1, self.nchannel+1):
            points[i] = self.baseline_model.baseline_correction(points[i])

    def trigEvent_baseline_corrected(self, i):
        points, trig = self.triggeredEvent(i)
        if trig : self.cnn_baseline_correction(points)
        return points, trig

    def baseline_check(self, i, n):
        points0 = self.getEventAdjusted(i)
        points1 = self.getEventAdjusted(i)
        self.cnn_baseline_correction(points1)
        plt.plot(points0[0], points0[n], label = 'original')
        plt.plot(points0[0], points1[n],label = 'corrected')
        line = np.linspace(points0[0][0], points0[0][-1], 100)
        zeros = np.zeros(100)
        plt.plot(line, zeros, '--',color='red',label='baseline')
        plt.legend(loc='best')

    def showWaveForm(self, channel):
        fig, ax1 = plt.subplots(dpi=180)
        for i in tqdm(range(int(self.nevent))):
            points, trigbit= self.triggeredEvent(i)
            if trigbit: 
                plt.plot(self.t, points[channel])
        ax1.set(xlabel='time (ps)', ylabel='amplitude',
                title='Wave Form of channel '+str(channel))
        #plt.text(self.t[1], ax1.get_ylim()[0], self.file_name, color='red')
        #plt.show()

    def showWaveForm_baseline_corrected(self, channel):
        fig, ax1 = plt.subplots(dpi=180)
        for i in tqdm(range(int(self.nevent))):
            points, trigbit= self.trigEvent_baseline_corrected(i)
            if trigbit: 
                plt.plot(self.t, points[channel])
        ax1.set(xlabel='time (ps)', ylabel='amplitude',
                title='Wave Form of channel '+str(channel))
        plt.show()

    def dist_waveform_integration(self, channel,number = -1, show = False):
        chg = []
        for i in tqdm(range(int(self.nevent))):
            charge = 0
            points, tig = self.triggeredEvent(i)
            y0 = points[channel][0:number]
            t0 = points[0][0:number]
            if tig: chg.append(np.trapz(y0,x = t0))
        if show:
            plt.hist(chg, bins = 40)
            plt.show()
        return chg

    def dist_waveform_integration_bkgSub(self, channel,number = -1, show = False):
        chg = []
        bkg = []
        for i in tqdm(range(int(self.nevent))):
            charge = 0
            points, tig = self.triggeredEvent(i)
            mean = (points[channel][self.bkg[0]:self.bkg[1]]).mean()
            t0 = points[0][self.bkg[0]:self.bkg[1]]
            if tig: 
                y1 = np.subtract(points[channel], mean)
                bkg.append(np.trapz(y1[self.bkg[0]:self.bkg[1]],x = self.t[self.bkg[0]:self.bkg[1]]))
                chg.append(np.trapz(y1[0:number],x = self.t[0:number]))
        if show:
            plt.hist(chg, bins = 40)
            plt.show()
        return chg, bkg


    def dist_waveform_integration2(self, channel, callf, show = False):
        chg = []
        for i in tqdm(range(int(self.nevent))):
            charge = 0
            points, tig = callf(i)

            y = points[channel]
            if tig: chg.append(np.trapz(y,x = points[0]))
        if show:
            plt.hist(chg, bins = 40)
            plt.show()
        return chg

        
