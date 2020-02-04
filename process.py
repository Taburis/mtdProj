
from mtdScope import scopeEmulator
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages

#class analyzer_base(list):
#    def __init__(self):
#        list.__init__(self)


class analyzer:
    def __init__(self,name, **kwargs):
        self.__name__ = name
        self.kwargs = kwargs
        self.data = []

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


class process:
    def __init__(self, name):
        self.__name__ = name
        self.scope = scopeEmulator()
        self.sequence = []
        self.endSequence = []
            
    def run(self):
        for f in self.flist:
            self.scope.loadData(f)
            self.scope.sliceEvent()
            for ana in self.sequence:
                for i in tqdm(range(int(self.scope.nevent))):
                    points, trig = self.scope.triggeredEvent(i)
                    if not trig: continue
                    ana.data = points
                with PdfPages(f+'.pdf') as pdf:
                    for ana in self.sequence:
                        ana.pdf = pdf
                        ana.end()
            with PdfPages(self.__name__+'_summary.pdf') as pdf:
                for ana in self.sequence:
                    pass

    
    def path(self, array):
        self.sequence = []
        for ele in list(array):
            self.sequence.append(ele)

    def source(self, flist):
        self.flist = flist

    def endPath(self, array):
        for ele in array:
            self.endSequence.append(ele)

