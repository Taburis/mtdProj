
from mtdScope import scopeEmulator
import matplotlib.pyplot as plt
import numpy as np
import os
import re

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

class mtdProcess:
    def __init__(self, infile):
        self.scope = scopeEmulator()
        self.scope.loadData(infile)
        self.scope.sliceEvent()
        self.output = ''
        self.buffer = {}
        self.transimpedence = -44000

    def charge_conversion(self, channel, bkg = [0, 1000]):
        self.scope.bkg = bkg
        chg, bkg = self.scope.dist_waveform_integration_bkgSub(channel)
        self.buffer['bkg_check_channel1'] = bkg
        self.buffer['charge_channel1'] = np.multiply(chg, self.transimpedence*1e-11*1e15)

