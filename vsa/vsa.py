from abc import ABCMeta, abstractmethod
import numpy as np

class VSA:
    
    mapping = {}                    # List of known mappings.
    size = 512                      # Length of memory vectors.
    distance_threshold = 0.20       # Distance from which similarity is accepted.
    
    verbose = False                 # Console verbose output.
    visualize = False               # Graph plotting of various steps of operations.
    
    @abstractmethod
    def __init__(self, input_value, memory=None):
        pass    
    
    ## Method that wipes previous mappings
    #
    #  @param self The object pointer.
    @classmethod
    def reset_kernel(self):
        VSA.mapping = {}   
    
    ## Setter for the length of symbolic vectors to be used.
    #  
    #  Every time this method is run all the previous mappings will be wiped.
    #
    #  @param self The object pointer.
    #  @param size The new size. Must be > 0.
    @classmethod
    def set_size(self, size):
        assert(size > 0)
        VSA.size = size
        VSA.reset_kernel() 
        
    @abstractmethod
    def __mul__(self, operand):
        pass   
    
    @abstractmethod
    def __sub__(self, operand):
        pass   
    
    @abstractmethod
    def __add__(self, operand):
        pass    
    
    @abstractmethod
    def __mod__(self, operand):
        pass  
    
    @abstractmethod
    def __div__(self, operand):
        pass   
    
    def encode(self, input_value):
        
        if not isinstance(input_value, np.ndarray) and input_value in VSA.mapping:
            return VSA.mapping[input_value]
        else:
            result = self.generateVector() 
            VSA.mapping[input_value] = result
            if self.visualize:    
                print("Encoded ", input_value)    
                self.plot(result)    
            return result
    
    @abstractmethod
    def generateVector(self):
        pass
    
    def decode(self, memory=None):
        
        result = []
        
        if memory == None:
            memory = self.memory
        
        for key in VSA.mapping:
            dist = self.distance(VSA.mapping[key], memory)
            if VSA.verbose :
                print("Distance from {} is {}".format(key,dist))
      
            if dist > self.distance_threshold:
                result.append((key, dist))
               
        return result 
    
    @abstractmethod
    def bind(self, one, other):
        pass  
    
    @abstractmethod
    def probe(self, one, other):
        pass    
    
    @abstractmethod
    def distance(self, one, other):
        pass
    
    ## A flexible plotting function that displays symbolic vectors graphically.
    #
    #  It can either be applied on an instance directly or on a given vector.
    #  Optionally it can reverse permute and smooth the vector prior to plotting.
    #  On demand a wider representation of the plotting window can be chosen.
    #
    #  @param self The object pointer.
    #  @param vect The vector to be plotted, in case it is not wished to plot a concrete instance.
    #  @param unpermute Boolean that prompts a reverse permute operation prior to plotting
    #  @param smooth Boolean that prompts a smoothing operation prior to plotting.
    #  @param unpermute Boolean that widens the margins of the displayed plotting window.
    #  @return The resulting permuted vector.     
    def plot(self, vect=None, wide=False):
        if vect is None:
            vect = self.memory
        plt.figure()
        xx = range(len(vect))
        if wide:
            widen = len(vect) * 0.1 
            down = np.amin(vect)
            up = np.amax(vect)
            mean = np.amax(vect) - np.amin(vect)
            down -= mean * 0.1
            up += mean * 0.1
            plt.axis([-widen, len(vect) + widen, down, up])
        plt.plot(xx, vect)
        plt.show()