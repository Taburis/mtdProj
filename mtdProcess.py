
from mtdScope import scopeEmulator
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy import optimize
from scipy.interpolate import interp1d
from scipy import integrate
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
import process as xp
import sigLib 
from scipy.stats import moyal
from scipy.optimize import curve_fit

def extract_number(istring):
    z = re.search("\d+",istring)
    return int(z.group())

def makeList(regex,folder):
    if folder[-1] != '/': folder = folder+'/'
    files = [f for f in os.listdir(folder)if os.path.isfile(folder+f) and re.match(regex,f)]
    return files

def makeList_v0(regex, folder):
    if folder[-1] != '/': folder = folder+'/'
    files = [folder+f+'/'+f+'_0.hdf5' for f in os.listdir(folder)if re.match(regex,f)]
    return files

def run_list_CFT(regex, folder, labelRule, trigger, channel):
    if folder[-1] != '/': folder = folder+'/'
    files = makeList(regex, folder)
    print(files)
    xs = []
    ys = []
    for f in files:
        print("processing: ",f)
        x = labelRule(f)
        dts, std = getCFT(folder+f, trigger, channel)
        xs.append(x)
        ys.append(std[str(channel[0])])
    return xs, ys

class trigger:
    def __init__(self):
        self.trig = 0.05
        self.max = 0.25

    def saturation_filter(self, points, i):
        ps = points[i]
        if np.amax(np.absolute(ps))>self.max : return False
        return True

    def noise_filter(self, points, i):
        ps = points[i]
        if np.amax(ps) > self.trig : return False
        return True

    def trig_single_channel(self, points, i):
        ps = points[i]
        if np.amax(np.absolute(ps))<self.trig : return False
        elif np.amax(np.absolute(ps))>self.max : return False
        elif np.amax(ps) > self.trig : return False
        return True

    def trigger_abs_single(self, points, i):
        ps = points[i]
        if np.amax(np.absolute(ps))<self.trig : return False
#        elif np.amax(np.absolute(ps))>self.max : return False
        return True

    def trigger_all3(self, points):
        trig1 = self.trig_single_channel(points, 1)
        trig3 = self.trig_single_channel(points, 3)
        trig2 = self.trig_single_channel(points, 2)
        return trig1 and trig3 and trig2

    def trigger_all4(self, points):
        trig1 = self.trig_single_channel(points, 1)
        trig2 = self.trig_single_channel(points, 2)
        trig3 = self.trig_single_channel(points, 3)
        trig4 = self.trigger_abs_single (points, 4)
        return trig1 and trig2 and trig3 and trig4

    def trig_check(self, ps):
        if np.amax(np.absolute(ps))<self.trig : return False
        elif np.amax(np.absolute(ps))>self.max : return False
        elif np.amax(ps) > self.trig : return False
        return True

    def trigger_range_ch1(self, points):
        trig2 = True
        trig3 = True
        if self.trig_single_channel(points, 1):
            ps = points[1]
            x0 = np.argmax(np.abs(ps))
            nn = 100
            trig2 = self.trig_check(points[2][x0-nn:x0+nn])
            trig3 = self.trig_check(points[3][x0-nn:x0+nn])
        else: return False
        if self.trig_single_channel(points, 2): False
        if self.trig_single_channel(points, 3): False
        return trig2 and trig3


class charge_analyzer(xp.analyzer):
    def __init__(self, **kwargs):
        super(charge_analyzer,self).__init__('charge_analyzer', **kwargs)
        self.charge_mean = 0
        self.bkg_range = kwargs['bkg_range']
        self.transimp = kwargs['transimp']
        self.channel = kwargs['channel']
        self.moyal = moyal()
        for i in self.channel:
            setattr(self,'charge_ch'+str(i),[])
            setattr(self,'bkg_ch'+str(i),[])

    def run(self):
        points = self.data
        t = points[0]
        for i in self.channel:
            mean = (points[i][self.bkg_range[0]:self.bkg_range[1]]).mean()
            t0 = points[0][self.bkg_range[0]:self.bkg_range[1]]
            y1 = points[i]
            y1 = np.subtract(points[i], mean) 
            bkg0 = np.trapz(y1[self.bkg_range[0]:self.bkg_range[1]],x = t[self.bkg_range[0]:self.bkg_range[1]])
            chg0 = np.trapz(y1,x = t)
            sign = np.sign(y1[np.argmax(np.abs(y1))])
            getattr(self,'charge_ch'+str(i)).append(sign*chg0/self.transimp*1e2)
            getattr(self,'bkg_ch'+str(i))   .append(sign*bkg0/self.transimp*1e2)

    def end(self):
        for i in self.channel:
            self.keep('charge_ch'+str(i), getattr(self,'charge_ch'+str(i)))
            self.keep('bkg_ch'+str(i) , getattr(self,'bkg_ch'+str(i)))
            getattr(self,'charge_ch'+str(i)).clear()
            getattr(self,'bkg_ch'+str(i)).clear()


class jitter_analyzer(xp.analyzer):
    def __init__(self, **kwargs):
        super(jitter_analyzer,self).__init__('jitter_analyzer', **kwargs)
        self.timer = sigLib.cftiming()
        self.dt21 = []
        self.dt31 = []
        self.dt32 = []
        self.dt123 = []
        self.dt1 = []
        self.dt2 = []
        self.dt3 = []

    def run(self):
        self.jitter_beamSetup_v0(self.data)

    def get_cftiming(self, points, channel):
        time = []
        for i in channel:
            try: 
                time.append(self.timer.getCFTiming(t = points[0], y= points[i]))
            except: 
                raise ValueError('CFT error skipped the event')
        return time

    def jitter_beamSetup_v0(self, points):
        try: 
            time = self.get_cftiming(points, [1,2,3])
        except: 
            return
        self.dt21.append(time[1]-time[0])
        self.dt32.append(time[2]-time[1])
        self.dt31.append(time[2]-time[0])
        self.dt123.append((time[2]+time[0])/2-time[1])
        self.dt1.append(time[0])
        self.dt2.append(time[1])
        self.dt3.append(time[2])


    def end(self):
        sig21 = np.array(self.dt21).std()
        sig31 = np.array(self.dt31).std()
        sig32 = np.array(self.dt32).std()
        sig123 = np.array(self.dt123).std()
        self.keep('sig1', np.array(self.dt1).std())
        self.keep('sig2', np.array(self.dt2).std())
        self.keep('sig3', np.array(self.dt3).std())
        self.keep('dt21', self.dt21)
        self.keep('dt31', self.dt31)
        self.keep('dt32', self.dt32)
        self.keep('dt123', self.dt123)
        self.keep('sig21', sig21)
        self.keep('sig31', sig31)
        self.keep('sig32', sig32)
        self.keep('sig123', sig123)
        #self.keep('cal2', (sig32**2+sig21**2)**0.5/2)
        self.keep('cal2', ((sig21**2-sig31**2+sig32**2)/2)**0.5)

        self.dt21.clear()
        self.dt31.clear()
        self.dt32.clear()


class hist_dist_dqm(xp.dqm):
    def __init__(self,name, **kwargs):
        super(hist_dist_dqm,self).__init__(name, **kwargs)
        self.query_list = [kwargs['x']]
        self.overlay = True

    def run(self):
        if self.overlay : self.overlay_plot()
        else : self.single_plot()

    def overlay_plot(self):
        fig, ax = plt.subplots(dpi = 110)
        for key in getattr(self,self.kwargs['x']).keys():
            plt.hist(np.array(getattr(self,self.kwargs['x'])[key]), bins = 60, label=self.label(key), alpha = 0.5, density = True)
        plt.legend(loc= 'best')
        ax.set(xlabel =self.kwargs['xlabel'], ylabel =self.kwargs['ylabel'], title = self.kwargs['title'])
        plt.grid(True)
        self.keep_fig()

    def single_plot(self):
        for key in getattr(self,self.kwargs['x']).keys():
            fig, ax = plt.subplots(dpi = 110)
            plt.hist(np.array(getattr(self,self.kwargs['x'])[key]), bins = 60, label=self.label(key), alpha = 0.5, density = False)
            plt.legend(loc= 'best')
            ax.set(xlabel =self.kwargs['xlabel'], ylabel =self.kwargs['ylabel'], title = self.kwargs['title'])
            plt.grid(True)
            self.keep_fig()

class func_xlabel_dqm(xp.dqm):
    def __init__(self,name, **kwargs):
        super(func_xlabel_dqm,self).__init__(name, **kwargs)
        self.query_list = []
        self.dict = {}

    def add_variable(self, var, label):
        self.query_list.append(var)
        self.dict[var] = label

    def draw(self, var):
        xs = []
        ys = []
        for key in getattr(self,var).keys():
            xs.append(self.label(key))
            ys.append(getattr(self,var)[key][()])
        order = np.argsort(xs)
        xs = np.array(xs)[order.astype(int)]
        ys = np.array(ys)[order.astype(int)]
        plt.plot(xs, ys, marker='o', label = self.dict[var])

    def run(self):
        fig, ax = plt.subplots(dpi = 110)
        for key in self.query_list:
            self.draw(key)
        ax.set(xlabel =self.kwargs['xlabel'], ylabel =self.kwargs['ylabel'], title = self.kwargs['title'])
        plt.legend(loc='best')
        #plt.ylim(60,130)
        plt.grid(True)
        self.keep_fig()


def HV_label(istring):
    return re.search('HV\d+',istring).group(0)

def HV_xlabel(istring):
    return int(re.search('HV\d+',istring).group(0)[2:])


dqm_jitter_dt21 = hist_dist_dqm('dqm_jitter_dt21', x='dt21', title='TOA of tch2-tch1',xlabel='TOA (ps)', ylabel = 'Fraction (%)')
dqm_jitter_dt21.label = HV_label
dqm_jitter_dt32 = hist_dist_dqm('dqm_jitter_dt32', x='dt32', title='TOA of tch3-tch2',xlabel='TOA (ps)', ylabel = 'Fraction (%)')
dqm_jitter_dt32.label = HV_label
dqm_jitter_dt31 = hist_dist_dqm('dqm_jitter_dt31', x='dt31', title='TOA of tch3-tch1',xlabel='TOA (ps)', ylabel = 'Fraction (%)')
dqm_jitter_dt31.label = HV_label
dqm_jitter_dt123 = hist_dist_dqm('dqm_jitter_dt123', x='dt123', title='TOA of (tch3+tch1)/2-tch2',xlabel='TOA (ps)', ylabel = 'Fraction (%)')
dqm_jitter_dt123.label = HV_label
dqm_jitter_dts = dqm_jitter_dt21+dqm_jitter_dt32+dqm_jitter_dt31+dqm_jitter_dt123

dqm_jitter = func_xlabel_dqm('dqm_cal2', title =  'Jitter vs Voltage', xlabel ='HV (V)', ylabel='Resolution (ps)')
dqm_jitter.add_variable('sig31', 'tch3-tch1')
dqm_jitter.add_variable('sig21', 'tch2-tch1')
dqm_jitter.add_variable('sig32', 'tch3-tch2')
dqm_jitter.add_variable('sig123', '(tch3+tch1)/2-tch2')
dqm_jitter.add_variable('cal2', 'drived ch2')
dqm_jitter.label = HV_xlabel

dqm_jitter_eachCh = func_xlabel_dqm('dqm_sig', title =  'Jitter vs Voltage', xlabel ='HV (V)', ylabel='Resolution (ps)')
dqm_jitter_eachCh.add_variable('sig1', 'tch1')
dqm_jitter_eachCh.add_variable('sig2', 'tch2')
dqm_jitter_eachCh.add_variable('sig3', 'tch3')
#dqm_jitter.add_variable('sig2', 'sig2')
dqm_jitter_eachCh.label = HV_xlabel

an_jitter_v0 = jitter_analyzer()
an_charge_v0 = charge_analyzer(channel=[1,2,3],transimp = 4400, bkg_range=[0, 400])
dqm_charge_ch1= hist_dist_dqm('dqm_charge', x= 'charge_ch1', title = 'charge distribution: channel1', xlabel ='charge (fC)', ylabel='Occurrence')
dqm_charge_ch1.label = HV_label
dqm_charge_ch1.overlay = False
dqm_charge_ch2= hist_dist_dqm('dqm_charge', x= 'charge_ch2', title = 'charge distribution: channel2', xlabel ='charge (fC)', ylabel='Occurrence')
dqm_charge_ch2.label = HV_label
dqm_charge_ch2.overlay = False
dqm_charge_ch3= hist_dist_dqm('dqm_charge', x= 'charge_ch3', title = 'charge distribution: channel3', xlabel ='charge (fC)', ylabel='Occurrence')
dqm_charge_ch3.label = HV_label
dqm_charge_ch3.overlay = False

dqm_charge = dqm_charge_ch1+dqm_charge_ch2+dqm_charge_ch3

