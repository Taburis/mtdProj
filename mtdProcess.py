
from mtdScope import scopeEmulator
import matplotlib.pyplot as plt
import numpy as np
import os
import re

def getCFT(infile, trigger, channels, nrep = 1000):
    se = scopeEmulator()
    se.loadData(infile)
    se.sliceEvent()
    ts = se.runTimeWalk(r0=0,r1=se.nevent, nstep = nrep, method='linear')
    std = {}
    dts = []
    for t in channels:
        dts.append(np.trim_zeros(np.subtract(ts[t-1],ts[trigger-1])))
        std[str(t)] = dts[-1].std()
    return dts, std

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
