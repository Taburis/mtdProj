
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
from sigProcess import cftiming
from scipy.stats import moyal
from scipy.optimize import curve_fit

def extract_number(istring):
    z = re.search("\d+",istring)
    return int(z.group())

def makeList(regex,folder):
    if folder[-1] != '/': folder = folder+'/'
    files = [f for f in os.listdir(folder)if os.path.isfile(folder+f) and re.match(regex,f)]
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

class trigger_book:
    def __init__(self):
        self.trig = 0.03
        self.max = 0.95

    def trig_single_channel(self, points, i):
        ps = points[i]
        if np.amax(np.absolute(ps))<trig : return False
        elif np.amax(np.absolute(ps))>trig_max : return False
        elif np.amax(ps) > 0.05 : return False
        return True

    def trigger_ch13(points):
        pps = points[1:]
        trig1 = False
        for ps in pps:
            trig1 = self.trig_single_channel(ps, 1)
            trig3 = self.trig_single_channel(ps, 3)
        return trig1 and trig3

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
        t = points[0]
        for i in self.channel:
            mean = (points[i][self.bkg_range[0]:self.bkg_range[1]]).mean()
            t0 = points[0][self.bkg_range[0]:self.bkg_range[1]]
            y1 = np.subtract(points[i], mean) 
            bkg0 = np.trapz(y1[self.bkg_range[0]:self.bkg_range[1]],x = t[self.bkg_range[0]:self.bkg_range[1]])
            chg0 = np.trapz(y1,x = t)
            sign = np.sign(np.amax(points[i]))
        chg.append(sign*chg0/self.transimp*1e2)
        bkg.append(sign*bkg0/self.transimp*1e2)

    def end(self):
        self.charge_mean = np.array(self.chg).mean()
        fig, ax = plt.subplots(num = self.__name__, dpi = 120)
        plt.hist(self.chg, bins =60, label = '')
#        plt.fit 
        ax.set(xlabel = 'charge (fC)', ylabel = 'Occurance', title='charge distribution')
        plt.legend(loc='best')


class jitter_analyzer(xp.analyzer):
    def __init__(self, **kwargs):
        super(jitter_analyzer,self).__init__('jitter_analyzer', **kwargs)
        self.timer = cftiming()
        self.dt21 = []
        self.dt31 = []
        self.dt32 = []
        self.sig31= []
        self.sig21= []
        self.sig32= []
        self.sig2 = []

    def run(self):
        self.jitter_beamSetup_v0(self.data)

    def get_cftiming(points, channel, timer):
        time = []
        for i in channel:
            time.append(timer.getCFTiming(t = points[0], y= points[i]))
        return time

    def jitter_beamSetup_v0(points):
        time = self.get_cftiming(points, [1,2,3], self.timer)
        self.dt21.append(time[1]-time[0])
        self.dt32.append(time[2]-time[1])
        self.dt31.append(time[2]-time[0])

    def end(self):
        self.sig21.append(np.array(self.dt21).std())
        self.sig31.append(np.array(self.dt31).std())
        self.sig32.append(np.array(self.dt32).std())
        self.sig2.append((self.sig21[-1]+self.sig31[-1]+self.sig32[-1])/2)

    def flash(self):
        self.dt21 = []
        self.dt31 = []
        self.dt32 = []
        self.sig31= []
        self.sig21= []
        self.sig32= []
        self.sig2 = []

    def summary(self):
        pass
