
import numpy as np
import scipy as sp
import scipy.ndimage
from scipy.signal import argrelextrema
from numpy.fft import fft, ifft
from numpy.linalg import norm
from numpy import array, sqrt, dot
import random
import matplotlib.pyplot as plt

from vsa import VSA
    
class HRR(VSA):

    mapping = {}  # list of known mappings so far
    permutation = {} # indexed by array size
    size = 256 #length of memory vectors
    stddev = 0.02 #standard deviation of gaussian bells for scalar encoders
    input_range = np.array([0,100]) # range of accepted inputs
    
    verbose = True
    visualize = False
    
    def __init__(self, v, size=None, memory=None, generator=None):
        
        if size == None:
            size = HRR.size               
        
        self.size = size
        self.label = v
        
        if memory != None:
            self.memory = memory 
        elif generator != None:
            memory = generator[0]
            for i in range(self.size):
                for j in range(len(generator) - 1):
                    memory[i] += generator[j+1][i]
                memory[i] /= len(generator)
            self.memory = memory
        else :
            self.memory = self.encode(size, v)    
    
    @classmethod
    def reset_kernel(self):
        HRR.mapping = {}     
        
    @classmethod
    def set_size(self, size):
        assert(size > 0)
        HRR.size = size
        create_permutation(size)
          
    def __add__(self, op):
        if op.__class__ != self.__class__:
            op = HRR(op)
        
        return HRR('-', memory=self.memory + op.memory)

    def __sub__(self, op):
        if op.__class__ != self.__class__:
            op = HRR(op)
        return HRR('-', memory=self.memory - op.memory)       
        
    def __mul__(self, op):
        if op.__class__ != self.__class__:
            op = HRR(op)
            
        #perform binding
        memory = self.circconv(self.memory, op.memory)       
        
        return HRR('-', memory=memory)
    
    def __mod__(self, op):
        
        if op.__class__ != self.__class__:
            op = HRR(op)
        
        # perform unbinding
        op_dec = self.periodic_corr(self.memory, op.memory)
        
        return HRR('-', memory=op_dec)
        
    def __div__(self, op):
    
        if op.__class__ != self.__class__:
            op = HRR(op)        
    
        # perform unbinding
        op_dec = self.periodic_corr(self.memory, op.memory) 
        
        if self.visualize:
            print("Output:")
            self.plot(op_dec)
            
        # cleanup of noisy result by dictionary lookup
        max_sim = {}
        max_v = -1
        max_idx = -1
        for k in HRR.mapping: # for every known key...
            v = self.distance(HRR.mapping[k], op_dec) # determine distance
            if HRR.verbose :
                print("Distance from {} is {}".format(k,v))
            
            if max_idx == -1 or max_v < v:
                max_v = v
                max_idx = k
                
            if v > 0.1:
                max_sim[k] = v
        
        # if no matches have been found in the dictionary try scalar decoder
        if not max_sim:   
            if self.visualize:
                print("Output Reverse:")
                self.plot(self.reverse_permute(op_dec))
            op_dec = smooth(self.reverse_permute(op_dec),self.size/50)  
            if self.visualize:
                print("Output Smooth:")
                self.plot(op_dec)
            while np.max(op_dec) > 6 * abs(np.mean(op_dec)):
                max_sim[len(max_sim)] = int(self.reverse_scale(np.argmax(op_dec), len(op_dec)))
                compensate = self.scalar_encoder(self.reverse_scale(np.argmax(op_dec), len(op_dec)), len(op_dec))
                compensate[:] = [x * -abs(np.max(op_dec)) for x in compensate]     
                op_dec += compensate      
        
        
        if HRR.verbose:
            return max_sim
        else:
            if max_v < 0.01:
                return "result too noisy"
            else:
                return max_idx
        
    def encode(self, sz, op):
        if op in HRR.mapping:
            return HRR.mapping[op]
        else:

            result = np.empty(self.size, dtype=float)
         
            if type(op) == float or type(op) == int: 
                result = self.permute(self.normalize(self.scalar_encoder(op, self.size)))
            elif type(op) == tuple:
                result = self.permute(self.normalize(self.coordinate_encoder(op)))
            else:    
                result = array([random.gauss(0,1) for i in range(self.size)])
                HRR.mapping[op] = result
            if self.visualize:    
                print("Encoded ", op)    
                self.plot(result)    
            return result
    
    def scalar_encoder(self,x,length):       
        
        result = np.empty(length, dtype=float)
        
        for i in range(length):
            result[i] = self.gaussian(i,self.scale(x, length),self.stddev * length) 
            
        return result
        
    
    def coordinate_encoder(self, x):   
        
        # get number of coordinates
        n = len(x)
        
        # compute individual lengths
        L = int(HRR.size / n)
        L_last = HRR.size - L * (n - 1)
        
        out = []
        for i in range(n-1):
            enc = self.scalar_encoder(x[i], L)
            out.extend(enc)
        # encode the last segment
        enc = self.scalar_encoder(x[n-1], L_last)
        out.extend(enc)
        
        return out
    
    def normalize(self,v):
        v -= np.sum(v)/self.size
        #assert(self.is_normalized(v))
        return v
    
    def is_normalized(self,v):
        return (abs(norm(v) - 1.0) <= 0.000001)
    
    def periodic_corr(self, x, y):  
        return ifft(fft(x) * fft(y).conj()).real
    
    def circconv(self, a, b):
        return np.real(ifft(fft(a)*fft(b)))
        
    def compare(self,one, other): # other is nparray
        scale=norm(one)*norm(other)
        if scale==0: return 0
        return dot(one,other)/(scale)
    
    def distance(self, one, other): #other is nparray
        return self.compare(one, other)
    
    def gaussian(self, x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    @classmethod
    def scale(self,x,L = None):
        return float(x - self.input_range[0]) * L / (self.input_range[1] - self.input_range[0])
    
    @classmethod
    def reverse_scale(self,x,L):
        return float(x - self.input_range[0]) / L * (self.input_range[1] - self.input_range[0])
    
    @classmethod
    def permute(self,x):
        n = len(x)
        if n not in self.permutation:
            create_permutation(n)
        result = np.empty(n, dtype=float)
        p = self.permutation[n]
        for i in range(n): 
            result[i] = x[p[i]]
        return result
    
    @classmethod
    def reverse_permute(self,x):
        n = len(x)
        p = self.permutation[n]
        result = np.empty(n, dtype=float)
        for i in range(n):
            result[p[i]] = x[i]
        return result
    
    @classmethod
    def plot(self,v):
        plt.figure()
        xx = range(len(v))
        plt.plot(xx, v)
        plt.show()
       
    @classmethod
    def plot2(self,v):
        plt.figure()
        xx = range(len(v))
        plt.plot(xx, v)
        plt.axis([-50,550,-6,6])
        plt.show()
        
    def decode(self):
        s = smooth(self.memory, window_length = self.size * 0.2)
        return np.argmax(s)
        s = smooth(HRR.reverse_permute(self.memory), window_len = self.size * 0.01)
        p = np.argmax(s)
        #p = np.argmax(HRR.reverse_permute(self.memory))
        #print('Before: {}'.format(p))
        p = (p / float(self.size)) * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
        #print('After: {}'.format(p))
        return p    
    
def create_permutation(L):
    HRR.permutation[L] = np.random.permutation(L)

create_permutation(HRR.size)

def smooth(x,window_len=100,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


