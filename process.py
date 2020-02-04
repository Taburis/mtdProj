class analyzer_wrapper:
    def __init__(self, func ,**kwargs):
        self.arg ={}
        self.exe= func
        for key, var in kwargs.items():
            self.arg[key]=var
           
    def run(self):
        f= self.exe
        f(**self.arg)

class analyzer(list):
    def __init__(self, func ,**kwargs):
        list.__init__(self)
        self.append(analyzer_wrapper(func, **kwargs))
    def run(self):
        for ana in self:
            ana.run()

class path(list):
    def __init__(self, array):
        list.__init__(self)
        for ele in array:
            self.append(ele)
            
    def __add__(self, rhs):
        for ele in array:
            self.append(ele)
    def run(self):
        for ana in self:
            ana.run()

