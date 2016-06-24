from abc import ABCMeta, abstractmethod

class VSA:
    
    @abstractmethod
    def __init__(self, v, size=None, memory=None):
        pass    
    
    @abstractmethod
    def reset_kernel(self):
        pass    
    
    @abstractmethod
    def __mul__(self, op):
        pass   
    
    @abstractmethod
    def __sub__(self, op):
        pass   
    
    @abstractmethod
    def __add__(self, op):
        pass    
    
    @abstractmethod
    def __mod__(self, op):
        pass  
    
    @abstractmethod
    def __div__(self, op):
        pass   
    
    @abstractmethod
    def encode(self, sz, op):
        pass  
    
    @abstractmethod
    def coordinate_encoder(self, x, limits_min, limits_max):
        pass 
    
    @abstractmethod
    def periodic_corr(self, x, y):
        pass   
    
    @abstractmethod
    def circconv(self, a, b):
        pass   
    
    @abstractmethod
    def compare(self,one, other):
        pass
    
    @abstractmethod
    def distance(self, one, other):
        pass
