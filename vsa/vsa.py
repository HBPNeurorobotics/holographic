from abc import ABCMeta, abstractmethod
import random
import numpy as np
import helpers
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class VSA(object):
    
    mapping = {}                    # List of known mappings.
    size = 512                      # Length of memory vectors.
    distance_threshold = 0.20       # Distance from which similarity is accepted.
    valid_range = zip([0],[100])    # Standard valid range for scalar or coordinate encoding
    return_list = False             # Boolean definind whether or not a list of results is returned, rather than just one.
    
    verbose = False                 # Console verbose output.
    visualize = False               # Graph plotting of various steps of operations.
    widening = 0.1                  # Ratio by which plotting windows should be widened 
    
    ## The constructor.
    #
    #  @param self The object pointer.
    #  @param input_value The value for which the VSA is constructed. May be of different types.
    #  @param memory A float vector which is the symbol by which the input value will be represented.
    #  @param generator A vector of diverse symbolic vectors, over which a mean is calculated and used as memory.
    def __init__(self, input_value, memory=None, generator=None, valid_range=None):
       
        self.label = input_value

        if valid_range != None:
            VSA.valid_range=valid_range
            
        if memory is not None:
            self.memory = memory 
        elif generator is not None:
            memory = generator[0]
            for i in range(self.size):
                for j in range(len(generator) - 1):
                    memory[i] += generator[j+1][i]
                memory[i] /= len(generator)
            self.memory = memory
        else:
            self.memory = self.encode(input_value, valid_range) 
    
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
            
        result = self.generateVector()
        VSA.mapping[input_value] = result 
        if self.visualize:    
            print("Encoded ", input_value)    
            self.plot(result) 
        return result
            
    ## Generates a normalized vector of random values.
    #
    #  @param self The object pointer.
    #  @return The generated vector.  
    def generateVector(self):
        return helpers.normalize(np.array([random.gauss(0,1) for i in range(self.size)]))
    
    def decode(self, memory=None):
        
        relist = []
        result = None
      
        if memory is None:
            memory = self.memory
        
        maxt = 0
        
        for key in VSA.mapping:
            dist = self.distance(VSA.mapping[key], memory)
            
            
            if self.verbose :
                print("Distance from {} is {}".format(key,dist))
                
            if dist > maxt:
                maxt = dist
                result = key
      
            relist.append((key, dist))
               
        if self.return_list:
            return relist
        else:
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
    #  2D vectors can also be visualized.
    #
    #  @param self The object pointer.
    #  @param vect The vector to be plotted, in case it is not wished to plot that of the current instance.
    #  @param unpermute Boolean that prompts a reverse permute operation prior to plotting
    #  @param smooth Boolean that prompts a smoothing operation prior to plotting.
    #  @param wide Boolean that widens the margins of the displayed plotting window.
    #  @param multidim Boolean that switches between the 1D and 2D representation
    #  @return The resulting permuted vector.     
    def plot(self, vect=None, unpermute=False, smooth=False, wide=False, multidim = False):
        
        if vect is None:
            vect = self.memory
        if unpermute:
            vect = self.reverse_permute(vect)
        if smooth:
            vect = helpers.smooth(vect)
            
        fig = plt.figure()
        
        if wide:
            widen = len(vect) * widening
            down = np.amin(vect)
            up = np.amax(vect)
            mean = np.amax(vect) - np.amin(vect)
            down -= mean * widening
            up += mean * widening
            plt.axis([-widen, len(vect) + widen, down, up])
        
        if multidim:
            assert(len(vect.shape) < 3)
            if (len(vect.shape) == 1):
                vect = helpers.reShape(vect,2)
            X = np.arange(-len(vect)/2,len(vect)/2,1)
            Y = np.arange(-len(vect[0])/2,len(vect[0])/2,1)
            X, Y = np.meshgrid(X, Y)
            ax = fig.gca(projection='3d')
            surf = ax.plot_surface(X, Y, vect, rstride=1, cstride=1, cmap='coolwarm', linewidth=0, antialiased=True)
            ax.set_zlim(np.min(vect)/3, 1.1*np.max(vect))
            ax.set_xlabel('Y Index')
            ax.set_ylabel('X Index')
            ax.set_zlabel('Encoded Value')
            ax.set_xlim3d(-len(vect)/2,len(vect)/2)
            ax.set_ylim3d(-len(vect[0])/2,len(vect[0])/2)
            ax.azim = 200
            fig.colorbar(surf, shrink=0.5, aspect=5)
        else:
            xx = range(len(vect))
            plt.plot(xx, vect)
            
        #fig.savefig('temp.png', transparent=True)
        plt.show()
