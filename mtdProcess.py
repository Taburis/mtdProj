
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

    def trigger_all3(self, points):
        trig1 = self.trig_single_channel(points, 1)
        trig3 = self.trig_single_channel(points, 3)
        trig2 = self.trig_single_channel(points, 2)
        return trig1 and trig3 and trig2

class charge_analyzer(xp.analyzer):
    def __init__(self, **kwargs):
        super(charge_analyzer,self).__init__('charge_analyzer', **kwargs)
        self.chg = []
        self.bkg = []
        self.charge_mean = 0
        self.bkg_range = kwargs['bkg_range']
        self.transimp = kwargs['transimp']
        self.channel = kwargs['channel']
        self.moyal = moyal()

    def run(self):
        points = self.data
        t = points[0]
        for i in self.channel:
            mean = (points[i][self.bkg_range[0]:self.bkg_range[1]]).mean()
            t0 = points[0][self.bkg_range[0]:self.bkg_range[1]]
            y1 = np.subtract(points[i], mean) 
            bkg0 = np.trapz(y1[self.bkg_range[0]:self.bkg_range[1]],x = t[self.bkg_range[0]:self.bkg_range[1]])
            chg0 = np.trapz(y1,x = t)
            sign = np.sign(np.amax(points[i]))
        self.chg.append(sign*chg0/self.transimp*1e2)
        self.bkg.append(sign*bkg0/self.transimp*1e2)

    def end(self):
        self.keep('charge_dist', self.chg)
        self.keep('bkg_check', self.bkg)
        self.chg.clear()
        self.bkg.clear()


class jitter_analyzer(xp.analyzer):
    def __init__(self, **kwargs):
        super(jitter_analyzer,self).__init__('jitter_analyzer', **kwargs)
        self.timer = sigLib.cftiming()
        self.dt21 = []
        self.dt31 = []
        self.dt32 = []

    def run(self):
        self.jitter_beamSetup_v0(self.data)

    def get_cftiming(self, points, channel):
        time = []
        for i in channel:
            time.append(self.timer.getCFTiming(t = points[0], y= points[i]))
        return time

    def jitter_beamSetup_v0(self, points):
        time = self.get_cftiming(points, [1,2,3])
        self.dt21.append(time[1]-time[0])
        self.dt32.append(time[2]-time[1])
        self.dt31.append(time[2]-time[0])


    def end(self):
        sig21 = np.array(self.dt21).std()
        sig31 = np.array(self.dt31).std()
        sig32 = np.array(self.dt32).std()
        self.keep('dt21', self.dt21)
        self.keep('dt31', self.dt31)
        self.keep('dt32', self.dt32)
        self.keep('sig21', sig21)
        self.keep('sig31', sig31)
        self.keep('sig32', sig32)
        self.keep('sig2', (sig21+sig31+sig32)/2)

        self.dt21.clear()
        self.dt31.clear()
        self.dt32.clear()


class hist_dist_dqm(xp.dqm):
    def __init__(self,name, **kwargs):
        super(hist_dist_dqm,self).__init__(name, **kwargs)
        self.query_list = [kwargs['x']]

    def run(self):
        fig, ax = plt.subplots(dpi = 110)
        for key in getattr(self,self.kwargs['x']).keys():
            plt.hist(np.array(getattr(self,self.kwargs['x'])[key]), bins = 60, label=self.label(key), alpha = 0.5)
        plt.legend(loc= 'best')
        ax.set(xlabel =self.kwargs['xlabel'], ylabel =self.kwargs['ylabel'], title = self.kwargs['title'])
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
        plt.grid(True)
        self.keep_fig()


def HV_label(istring):
    return re.search('HV\d+',istring).group(0)

def HV_xlabel(istring):
    return int(re.search('HV\d+',istring).group(0)[2:])

dqm_charge = hist_dist_dqm('dqm_charge', x= 'charge_dist', title = 'charge distribution', xlabel ='charge (fC)', ylabel='Occurance')
dqm_charge.label = HV_label

dqm_jitter = func_xlabel_dqm('dqm_sig21', title =  'Jitter vs Voltage', xlabel ='HV (V)', ylabel='jitter (ps)')
dqm_jitter.add_variable('sig31', 'tch3-tch1')
dqm_jitter.add_variable('sig21', 'tch2-tch1')
dqm_jitter.add_variable('sig32', 'tch3-tch2')
#dqm_jitter.add_variable('sig2', 'sig2')
dqm_jitter.label = HV_xlabel

an_jitter_v0 = jitter_analyzer()
an_charge_v0 = charge_analyzer(channel=[2],transimp = -4400, bkg_range=[0, 400])

