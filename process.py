
from mtdScope import scopeEmulator
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
import h5py
import copy

#class analyzer_base(list):
#    def __init__(self):
#        list.__init__(self)


class analyzer:
    def __init__(self,name, **kwargs):
        self.__name__ = name
        self.kwargs = kwargs
        self.data = []
        self.buffer = {}

    def run(self):
        pass

    def __add__(self, rhs):
        res = []
        res.append(self)
        if isinstance(rhs,list): 
            for i in rhs:
                res.append(i)
        else : res.append(rhs)
        return res

    def __radd__(self, lhs):
        res = []
        if isinstance(lhs,list): 
            for i in lhs:
                res.append(i)
        else : res.append(lhs)
        res.append(self)
        return res

    def keep(self, key, data):
        self.buffer[key] = copy.copy(data)

class dqm(analyzer):
    def __init__(self, name, **kwargs):
        super(dqm, self).__init__(name, **kwargs)
        self.query_list = []

    def link(self, process):
        self.processor = process
        self.canvas = process.dqm
        for x in self.query_list:
            setattr(self, x, {})

    def query(self, group):
        data = self.processor.output_file[group]
        for key in self.query_list:
            getattr(self,key)[group]= data[key] 

    def keep_fig(self):
        self.processor.dqm.savefig()

class process:
    def __init__(self, name):
        self.__name__ = name
        self.scope = scopeEmulator()
        self.sequence = []
        self.endSequence = []
        self.buffer_group = []
            
    def run(self):
        self.output_file = h5py.File(self.__name__+'_output.h5','w')
        for f in self.flist:
            f_name = f.split('/')[-1].split('.')[-2]
            self.scope.loadData(f)
            self.scope.sliceEvent()
            for i in tqdm(range(int(self.scope.nevent))):
                points, trig = self.scope.triggeredEvent(i)
                for ana in self.sequence:
                    if not trig: continue
                    ana.data = points
                    ana.run()
            for ana in self.sequence:
                ana.end()
                self.load_buffer(f_name, ana)
            self.buffer_group.append(f_name)
        self.dqm = PdfPages(self.__name__+'_dqm.pdf')
        for dqm in self.dqmSequence:
            dqm.link(self)
            # a group is a input file
            for group in self.buffer_group:
                dqm.query(group)
            dqm.run()
        self.dqm.close()
        self.output_file.close()
    
    def path(self, array):
        self.sequence = []
        if isinstance(array,list): 
            for ele in list(array):
                self.sequence.append(ele)
        else : self.sequence.append(array)

    def source(self, flist):
        self.flist = flist

    def load_buffer(self, label, ana):
        for key in ana.buffer.keys():
#            print(key,ana.buffer[key])
            self.output_file.create_dataset(label+'/'+key,data = copy.copy(ana.buffer[key]))
        ana.buffer.clear()

    def endPath(self, array):
        self.endSequence = []
        if isinstance(array,list): 
            for ele in list(array):
                self.endSequence.append(ele)
        else : self.endSequence.append(array)

    def waveform_check(self,channel):
        self.scope.loadData(self.flist[0])
        self.scope.sliceEvent()
        self.scope.showWaveForm(channel)

    def dqmPath(self,array):
        self.dqmSequence = []
        if isinstance(array,list): 
            for ele in list(array):
                self.dqmSequence.append(ele)
        else : self.dqmSequence.append(array)


