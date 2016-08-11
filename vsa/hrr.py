import numpy as np
import scipy as sp
import scipy.ndimage
from scipy.signal import argrelextrema
from numpy.fft import fft, ifft
from numpy.linalg import norm
from numpy import array, sqrt, dot
import random
import matplotlib.pyplot as plt
import numbers

import helpers
from vsa import VSA

##  Holographic Reduced Representations.
# 
#   A Vector Symbolic Architecture that allows the user to work with and 
#   perform operations on symbols, which hold the place of various inputs.

class HRR(VSA):

    mapping = {}                    # List of known mappings.
    permutation = {}                # List of vector permutations, indexed by vector size.
    size = 256                      # Length of memory vectors.
    stddev = 0.02                   # Standard deviation of gaussian bells for scalar encoders.
    input_range = np.array([0,100]) # Range of accepted inputs.
    distance_threshold = 0.20       # Distance from which similarity is accepted.
    incremental_weight = 0.1        # Weight used for induction.
    peak_min = 10                    # Minimum factor by which a gaussian peak needs be larger than the average of noise.
    
    verbose = True                  # Console verbose output.
    visualize = False               # Graph plotting of various steps of operations.
    
    ## The constructor.
    #
    #  @param self The object pointer.
    #  @param input_value The value for which the HRR is constructed. May be of different types.
    #  @param memory A float vector which is the symbol by which the input value will be represented.
    #  @param generator A vector of diverse symbolic vectors, over which a mean is calculated and used as memory.
    def __init__(self, input_value, memory=None, generator=None):
       
        self.label = input_value

        if memory is not None:
            self.memory = memory 
        elif generator is not None:
            memory = generator[0]
            for i in range(self.size):
                for j in range(len(generator) - 1):
                    memory[i] += generator[j+1][i]
                memory[i] /= len(generator)
            self.memory = memory
        else :
            self.memory = self.encode(input_value)    
    
    ## Setter for the length of symbolic vectors to be used.
    #  
    #  Every time this method is run all the previous mappings will be wiped.
    #
    #  @param self The object pointer.
    #  @param size The new size. Must be > 0.
    @classmethod
    def set_size(self, size):
        assert(size > 0)
        HRR.size = size
        HRR.reset_kernel()
    
    ## Method that wipes previous mappings
    #
    #  @param self The object pointer.
    @classmethod
    def reset_kernel(self):
        HRR.mapping = {}     
   
    ## Overload of the "+" operand.
    #
    #  Creates a new HRR containing the sum of the other two memories as its own.
    #
    #  @param self The object pointer.
    #  @param operand The second operand of the operation.
    #  @return The new HRR.
    def __add__(self, operand):
        
        if operand.__class__ != self.__class__:
            operand = HRR(operand)
        return HRR('-', memory=self.memory + operand.memory)

    ## Overload of the "-" operand.
    #
    #  Creates a new HRR containing the subtraction of the other two memories as its own.
    #
    #  @param self The object pointer.
    #  @param operand The second operand of the operation.
    #  @return The new HRR.
    def __sub__(self, operand):
        
        if operand.__class__ != self.__class__:
            operand = HRR(operand)
        return HRR('-', memory=self.memory - operand.memory)    
    
    ## Overload of the "*" operand.
    #
    #  Creates a new HRR containing by binding the other two with the help of circular convolution.
    #
    #  @param self The object pointer.
    #  @param operand The second operand of the operation.
    #  @return The new HRR.        
    def __mul__(self, operand):
        if operand.__class__ != self.__class__:
            operand = HRR(operand)
        memory = self.circconv(self.memory, operand.memory) #perform binding       
        
        return HRR('-', memory=memory)
    
    ## Overload of the "%" operand.
    #
    #  Creates a new HRR containing by probing the first with the second via periodic correlation.
    #
    #  @param self The object pointer.
    #  @param operand The second operand of the operation.
    #  @return The new HRR.    
    def __mod__(self, operand):
        
        if operand.__class__ != self.__class__:
            operand = HRR(operand)
        memory = self.periodic_corr(self.memory, operand.memory) #perform unbinding
        
        return HRR('-', memory=memory)
    
    ## Overload of the "/" operand.
    #
    #  Probes the first HRR with the second via periodic correlation and decodes the result, returning it as a dictionary.
    #  The value of the keys represents the distance to the actual mapping for non-scalars.
    #  For scalars the value is always 1. 
    #
    #  @param self The object pointer.
    #  @param operand The second operand of the operation.
    #  @return A dictionary of the decoded values. Can also be empty or contain more than one value.       
    def __div__(self, operand):
    
        if operand.__class__ != self.__class__:
            operand = HRR(operand)        
        memory = self.periodic_corr(self.memory, operand.memory) #perform unbinding
        
        if self.visualize:
            print("Output:")
            self.plot(memory)
        
        return self.decode(memory)
    
    ## Overload of the "**" operand.
    #
    #  Further induces an HRR's memory with that of another one.
    #
    #  @param self The object pointer.
    #  @param operand The second operand of the operation.
    #  @return The new HRR.     
    def __pow__(self, operand):
        
        if operand.__class__ != self.__class__:
            operand = HRR(operand)
            
        return HRR('-', memory = self.memory - self.incremental_weight * (self.memory - operand.memory))
    
    ## Decodes a symbol and retrieves its content.
    #
    #  The result is a dictionary, in which the keys represent the decoded result.
    #  The value of the keys represents the distance to the actual mapping for non-scalars.
    #  For scalars the value is always 1.
    #  The symbol is first matched against all existing mappings and only then treated as a scalar.
    #
    #  @param self The object pointer.
    #  @param memory The memory vector.
    #  @param return_dict Whether to return a dictionary of all values or just the first value.
    #  @param suppress_value The given value (if any) will be suppressed from the result.
    #  @return A dictionary containing the decoded content or the first one depending on suppress_value.
    def decode(self, memory=None, return_dict=False, suppress_value=None):
        
        if memory == None:
            memory = self.memory
        
        # Dictionary lookup for existing mapping.
        result = {}
        match = False
        for key in HRR.mapping:
            dist = self.distance(HRR.mapping[key], memory) # determine distance
            if HRR.verbose :
                print("Distance from {} is {}".format(key,dist))
      
            if dist > self.distance_threshold:
                result[key] = dist
                match = True
        
        # If no matches have been found in the dictionary try scalar decoder.
        if not match:   
            if self.visualize:
                print("Output Reverse:")
                self.plot(self.reverse_permute(memory))
            memory = helpers.smooth(self.reverse_permute(memory),self.size/50)  
            if self.visualize:
                print("Output Smooth:")
                self.plot(memory)
            if suppress_value is not None:
                # suppress the given value in the memory
                compensate = self.scalar_encoder(suppress_value, len(memory))
                compensate[:] = [x * -abs(np.max(memory)) for x in compensate]
                memory += compensate
            while np.max(memory) > self.peak_min * abs(np.mean(memory)):
                spot = helpers.reverse_scale(np.argmax(memory), len(memory), self.input_range)
                result[spot] = 1
                compensate = self.scalar_encoder(spot, len(memory))
                compensate[:] = [x * -abs(np.max(memory)) for x in compensate]     
                memory += compensate 
                if self.visualize:
                    print("Output Smooth:")
                    self.plot(memory)
                if return_dict == False:
                    return spot
        if len(result) == 0 and suppress_value is not None:
            return np.nan
        return result
    
    ## Creates an encoding for a given input.
    #
    #  For non-scalars, should the input have a previous mapping it will return it. New mappings are first stored.
    #  For scalars the encoding is not stored, since it is generated as a Gaussian Bell at the spot of the scalar
    #  which is permuted via a fixed permutation in order to facilitate lots of frequencies in Fourier space.
    #
    #  @param self The object pointer.
    #  @param input_value The input to be encoded.
    #  @return The encoded vector.     
    def encode(self, input_value):
        if input_value in HRR.mapping:
            return HRR.mapping[input_value]
        else:

            result = np.empty(self.size, dtype=float)
         
            if isinstance(input_value, float) or isinstance(input_value, numbers.Integral):
                result = self.permute(helpers.normalize(self.scalar_encoder(input_value, self.size)))
            elif type(input_value) == tuple:
                result = self.permute(helpers.normalize(self.coordinate_encoder(input_value)))
            else:    
                result = array([random.gauss(0,1) for i in range(self.size)])
                HRR.mapping[input_value] = result
            if self.visualize:    
                print("Encoded ", input_value)    
                self.plot(result)    
            return result
    
    ## Samples a Gaussian bell over a certain range, permutes and stores it in a vector.
    #
    #  In order to create lots of frequencies in Furier space all sampled Gaussian bells are permuted
    #  and all operations are run on them in this way. For plotting purposes it is necessary to mark
    #  the "unpermute" flag to True.
    #
    #  @param self The object pointer.
    #  @param scalar The scalar value that will be the peak of the Gaussian.
    #  @param length The desired length of the resulting vector.
    #  @return The encoded vector.  
    def scalar_encoder(self,scalar,length):       
        
        result = np.empty(length, dtype=float)
        
        for i in range(length):
            result[i] = helpers.gaussian(i,helpers.scale(scalar, length, self.input_range),self.stddev * length) 
            
        return result
        
    ## TODO
    #
    #  @param self The object pointer.
    #  @param coordinates The coordinates that will be encoded.
    #  @return The encoded vector.      
    def coordinate_encoder(self, coordinates):   
        
        # get number of coordinates
        nr = len(coordinates)
        
        # compute individual lengths
        length = int(HRR.size / nr)
        length_last = HRR.size - length * (nr - 1)
        
        out = []
        for i in range(nr-1):
            enc = self.scalar_encoder(coordinates[i], length)
            out.extend(enc)
        # encode the last segment
        enc = self.scalar_encoder(coordinates[n-1], length_last)
        out.extend(enc)
        
        return out
    
    ## Performs periodic correlation on two symbolic vectors.
    #
    #  @param self The object pointer.
    #  @param one The first of the two vectors.
    #  @param one The second vector.
    #  @return The result of periodic correlation as a vector.             
    def periodic_corr(self, one, other):  
        return ifft(fft(one) * fft(other).conj()).real
    
    ## Performs circular convolution on two symbolic vectors.
    #
    #  @param self The object pointer.
    #  @param one The first of the two vectors.
    #  @param one The second vector.
    #  @return The result of circular convolution as a vector.     
    def circconv(self, one, other):
        return np.real(ifft(fft(one)*fft(other)))
    
    ## Calculates the distance between two symbolic vectors
    #
    #  This distance is given by the dot product of the two vectors.
    #
    #  @param self The object pointer.
    #  @param one The first of the two vectors.
    #  @param one The second vector.
    #  @return The distance between the two vectors. 1 means the vectors match, 0 that they are orthogonal.     
    def distance(self,one, other):
        scale=norm(one)*norm(other)
        if scale==0: return 0
        return dot(one,other)/(scale)
    
    ## Permutes a symbolic vector according to a fixed predefined permutation, which adapts to the vector size.
    #
    #  @param self The object pointer.
    #  @param vect The vector that will be permuted.
    #  @return The resulting permuted vector. 
    def permute(self,vect):
        nr = len(vect)
        if nr not in self.permutation:
            HRR.permutation[nr] = np.random.permutation(nr)
        return helpers.permute(vect,HRR.permutation[nr])
 
    ## Reverse permutes a symbolic vector according to a fixed predefined permutation, which adapts to the vector size.
    #
    #  It is the exact opposite of the permute method, and will return the original vector if used on top of it.
    #
    #  @param self The object pointer.
    #  @param vect The vector that will be reverse permuted.
    #  @return The resulting reverse permuted vector. 
    def reverse_permute(self,vect):
        nr = len(vect)
        if nr not in self.permutation:
            HRR.permutation[nr] = np.random.permutation(nr)
        return helpers.reverse_permute(vect,HRR.permutation[nr])
    
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
    def plot(self, vect=None, unpermute=False, smooth=False, wide=False):
        if vect == None:
            vect = self.memory
        if unpermute:
            vect = self.reverse_permute(vect)
        if smooth:
            vect = helpers.smooth(vect)
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

#    def decode(self):
#        s = helpers.smooth(self.memory, window_length = self.size * 0.2)
#        return np.argmax(s)
#        s = helpers.smooth(HRR.reverse_permute(self.memory), window_len = self.size * 0.01)
#        p = np.argmax(s)
#        #p = np.argmax(HRR.reverse_permute(self.memory))
#        #print('Before: {}'.format(p))
#        p = (p / float(self.size)) * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
#        #print('After: {}'.format(p))
#        return p    



