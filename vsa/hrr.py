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

    permutation = {}                # List of vector permutations, indexed by vector size.
    stddev = 0.02                   # Standard deviation of gaussian bells for scalar encoders.
    incremental_weight = 0.1        # Weight used for induction.
    peak_min = 10                   # Minimum factor by which a gaussian peak needs be larger than the average of noise.
    
    ## The constructor.
    #
    #  @param self The object pointer.
    #  @param input_value The value for which the HRR is constructed. May be of different types.
    #  @param memory A float vector which is the symbol by which the input value will be represented.
    #  @param generator A vector of diverse symbolic vectors, over which a mean is calculated and used as memory.
    def __init__(self, input_value, memory=None, generator=None, valid_range=None):
       
        self.label = input_value

        if valid_range == None:
            valid_range = [0,100]
            
        self.set_valid_range(valid_range)

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

    def set_valid_range(self, valid_range):
        if valid_range is None:
            self.valid_range = valid_range
            return
        self.valid_range = valid_range

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
        memory = self.bind(self.memory, operand.memory) #perform binding       
        
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
        memory = self.probe(self.memory, operand.memory) #perform unbinding
        
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
        memory = self.probe(self.memory, operand.memory) #perform unbinding
        
        if self.visualize:
            print("Output:")
            self.plot(memory)
        
        return self.decode(memory, self.valid_range)
    
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
    #  The result is a list of tuples, in which the left values represent the decoded result.
    #  The right value of the tuples represent the distance to the actual mapping for non-scalars.
    #  For scalars the right value is always 1.
    #  The symbol is first matched against all existing mappings and only then treated as a scalar.
    #
    #  @param self The object pointer.
    #  @param memory The memory vector.
    #  @param return_list Whether to return a list of all values or just the first value.
    #  @param suppress_value The given value (if any) will be suppressed from the result.
    #  @return A dictionary containing the decoded content or the first one depending on return_list.
    def decode(self, memory=None, return_list=False, suppress_value=None, input_range=None, decode_range=None):
        
        if memory == None:
            memory = self.memory

        if decode_range is None:
            decode_range = self.valid_range
        if decode_range is None:
            raise ValueError("Decoding scalar values requires valid range (valid_range or decode_range parameter)")
        
        result = VSA.decode(self, memory)
        match = bool(result)
        
        # If no matches have been found in the dictionary try scalar decoder.
        if not match:   
            if self.visualize:
                print("Output Reverse:")
                self.plot(self.reverse_permute(memory))
            # check if elements (here: first) of decode_range are tuples -> multidimensional case
            if isinstance(decode_range[0], (frozenset, list, np.ndarray, set, tuple)):
                assert(False)
                # decoding tuple -> smooth individual chunks
                num_rng = len(decode_range) # number of ranges, e.g. 3 for 3D output
                for i, rng in enumerate(decode_range):
                    start_idx = int(i * (self.size / num_rng))
                    end_idx = int((i + 1) * (self.size / num_rng))
                    chunk_size = end_idx - start_idx
                    this_hrr = memory[start_idx:end_idx]
                    this_hrr = helpers.smooth(self.reverse_permute(this_hrr), self.size/50)[:chunk_size]
            else:
                # decoding single scalar value -> smooth complete memory
                memory = helpers.smooth(self.reverse_permute(memory), self.size/50)[:self.size]
            if self.visualize:
                print("Output Smooth:")
                self.plot(memory)

            if suppress_value is not None:
                #result = self.permute(helpers.normalize(self.coordinate_encoder(input_value, encode_range)))
                assert(input_range is not None)
                # suppress the given values in the memory
                if not isinstance(suppress_value, (frozenset, list, np.ndarray, set, tuple)):
                    suppress_value = [suppress_value]
                compensate = array(self.coordinate_encoder(suppress_value, input_range))
                # we have to smooth this to ensure correct alignment with the current memory
                compensate = helpers.smooth(compensate, self.size/50)[:self.size]
                if self.visualize:
                    self.plot(compensate)
                compensate[:] = [x * -abs(np.max(memory)) for x in compensate]
                memory += compensate
                #for i, v in enumerate(suppress_value):
                #    assert(len(suppress_value) == len(input_range))
                #    compensate = self.scalar_encoder(v, len(memory), input_range[i])
                #    compensate[:] = [x * -abs(np.max(memory)) for x in compensate]
                #    memory += compensate
                if self.visualize:
                    print("Output Smooth (after suppression):")
                    self.plot(memory)
            while np.max(memory) > self.peak_min * abs(np.mean(memory)):
                spot = []
                # check if elements (here: first) of decode_range are tuples -> multidimensional case
                if isinstance(decode_range[0], (frozenset, list, np.ndarray, set, tuple)):
                    assert(False)
                    # decoding tuple -> split memory in len(decode_range) chunks
                    num_rng = len(decode_range) # number of ranges, e.g. 3 for 3D output
                    for i, rng in enumerate(decode_range):
                        start_idx = int(i * (self.size / num_rng))
                        end_idx = int((i + 1) * (self.size / num_rng))
                        this_hrr = memory[start_idx:end_idx]
                        value = np.argmax(this_hrr)
                        spot.append(helpers.reverse_scale(value, int(HRR.size/num_rng), rng))
                else:
                    # decoding single scalar value (non-tuple)
                    spot = helpers.reverse_scale(np.argmax(memory), len(memory), decode_range)
                result.append((spot, 1))
                if return_list == False:
                    return spot
                if isinstance(decode_range[0], (frozenset, list, np.ndarray, set, tuple)):
                    compensate = self.coordinate_encoder(spot, decode_range)
                else:
                    compensate = self.scalar_encoder(spot, len(memory), decode_range)
                compensate[:] = [x * -abs(np.max(memory)) for x in compensate]     
                memory += compensate 
                if self.visualize:
                    print("Output Smooth:")
                    self.plot(memory)
        if len(result) == 0 and suppress_value is not None:
            return [(np.nan, 1)] if return_list else np.nan
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
    def encode(self, input_value, encode_range=None):
        if encode_range is None:
            encode_range = self.valid_range
        if encode_range is None:
            raise ValueError("Encoding scalar values requires valid range (valid_range or encode_range parameter)")

        if not isinstance(input_value, np.ndarray) and input_value in HRR.mapping:
            return HRR.mapping[input_value]
        else:

            result = np.empty(self.size, dtype=float)
         
            if isinstance(input_value, float) or isinstance(input_value, numbers.Integral):
                result = self.permute(helpers.normalize(self.scalar_encoder(input_value, self.size, encode_range)))
            elif isinstance(input_value, (frozenset, list, np.ndarray, set, tuple)):
                result = self.permute(helpers.normalize(self.coordinate_encoder(input_value, encode_range)))
            else:    
                result = self.generateVector() 
                HRR.mapping[input_value] = result
            if self.visualize:    
                print("Encoded ", input_value)    
                self.plot(result)    
            return result

    ## Generates a vector of random values.
    #
    #  @param self The object pointer.
    #  @return The generated vector.           
    def generateVector(self):
        return array([random.gauss(0,1) for i in range(self.size)])
    
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
    def scalar_encoder(self, scalar, length, scale_range):
        
        result = np.empty(length, dtype=float)
        
        for i in range(length):
            result[i] = helpers.gaussian(i, helpers.scale(scalar, length, scale_range), self.stddev * length)
            
        return result
        
    ## TODO
    #
    #  @param self The object pointer.
    #  @param coordinates The coordinates that will be encoded.
    #  @return The encoded vector.      
    def coordinate_encoder(self, coordinates, encode_range):
        
        # get number of coordinates
        nr = len(coordinates)
        assert(len(encode_range) == nr)
        
        # compute individual lengths
        length = int(HRR.size / nr)
        length_last = HRR.size - length * (nr - 1)
        
        out = []
        for i in range(nr-1):
            enc = self.scalar_encoder(coordinates[i], length, encode_range[i])
            out.extend(enc)
        # encode the last segment
        enc = self.scalar_encoder(coordinates[nr-1], length_last, encode_range[nr-1])
        out.extend(enc)
        
        return out
    
    ## Performs periodic correlation on two symbolic vectors.
    #
    #  @param self The object pointer.
    #  @param one The first of the two vectors.
    #  @param one The second vector.
    #  @return The result of periodic correlation as a vector.             
    def probe(self, one, other):  
        return ifft(fft(one) * fft(other).conj()).real
    
    ## Performs circular convolution on two symbolic vectors.
    #
    #  @param self The object pointer.
    #  @param one The first of the two vectors.
    #  @param one The second vector.
    #  @return The result of circular convolution as a vector.     
    def bind(self, one, other):
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
        if vect is None:
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



