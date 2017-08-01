import numpy as np
from numpy.linalg import norm

## Helper Methods.
#
#  A variety of methods that aid the functionality of Vector Symbolic Architectures.

## Calculates the value of a Gaussian at a certain point.
#
#  @param pos The position for which the value is requested.
#  @param mu The expected value of the Gaussian.
#  @param sig The variance of the Gaussian.
#  @return The value of a Gaussian at a certain point.
def gaussian(pos, mu, sig):
    return np.exp(-np.power(pos - mu, 2.) / (2 * np.power(sig, 2.)))

## Compute the sampled array of an 1-dimensional Gaussian
#
#  @param srange Range for input values [x_min, x_max]
#  @param num_samples Number of samples
#  @param mu The expected value of the Gaussian
#  @param sig The standard deviation of the Gaussian
#  @return The linearly sampled array of a Gaussian
def gaussian1d(srange, num_samples, mu, sig):
    x = np.linspace(srange[0], srange[1], num_samples)
    return np.exp(-np.true_divide(np.power(x - mu, 2.0), np.multiply(np.power(sig, 2.0), 2.0)))

## Compute the meshgrid of a 2-dimensional Gaussian
#
#  @param ranges List of ranges for input values [[x_min, x_max], [y_min, y_max]]
#  @param num_samples List of number of samples for each dimension [num_smpl_x, num_smpl_y]
#  @param mu1 The expected value of x-dimension
#  @param mu2 The expected value of y-dimension
#  @param sig1 The standard deviation of x-dimension
#  @param sig2 The standard deviation of y-dimension
#  @return The bi-linearly sampled meshgrid of a Gaussian
def gaussian2d(ranges, num_samples, mu1, mu2, sig1, sig2):
    x = np.linspace(ranges[0][0], ranges[0][1], num_samples[0])
    y = np.linspace(ranges[1][0], ranges[1][1], num_samples[1])
    xx, yy = np.meshgrid(x, y, sparse=True)
    x_val = np.true_divide(np.power(np.subtract(xx, mu1), 2.0), np.multiply(np.power(sig1, 2.0), 2.0))
    y_val = np.true_divide(np.power(np.subtract(yy, mu2), 2.0), np.multiply(np.power(sig2, 2.0), 2.0))
    return np.exp(-np.add(x_val, y_val))

## Normalizez a vector.
#
#  This is done in such a way that the norm of the vector becomes one
#  and the mean of the vector is zero.
#
#  @param vect The vector to be normalized.
#  @result The normalized vector.
def normalize(vect):
    vect -= np.sum(vect)/len(vect)
    vect /= norm(vect)
    return vect

## Permutes a vector using a given permutation.
#
#  @param vect The vector to be permuted.
#  @param perm The permutation to be followed.
#  @result The permuted vector.
def permute(vect, perm):
    l = len(vect)
    result = np.empty(l, dtype=float)
    for i in range(l): 
        result[i] = vect[perm[i]]
    return result

## Reverse permutes a vector using a given permutation.
#
#  @param vect The vector to be reverse permuted.
#  @param perm The reverse permutation to be followed.
#  @result The reverse permuted vector.
def reverse_permute(vect, perm):
    l = len(vect)
    result = np.empty(l, dtype=float)
    for i in range(l): 
        result[perm[i]] = vect[i]
    return result

## Transfers a value from an arbirtary input scale to a position in a vector.
#
#  The position in the vector is only positive. Due to this only the total length must be passed on.
#  The input scale on the other hand must pass its min and max values.
#
#  @param val The val to be scaled.
#  @param length The length of the vector.
#  @param input_range A vector containing only two positions: the min and max values of the input.
#  @result The scaled value.
def scale(val,length ,input_range):
    return float(val - input_range[0]) * length / (input_range[1] - input_range[0])

## Transfers a value from a position in a vector to an arbitrary scale.
#
#  It represents the inverse operation to the scale method.
#  The position in the vector is only positive. Due to this only the total length must be passed on.
#  The input scale on the other hand must pass its min and max values.
#
#  @param val The val to be scaled.
#  @param length The length of the vector.
#  @param input_range A vector containing only two positions: the min and max values of the input.
#  @result The scaled value.
def reverse_scale(val,length ,input_range):
    return float(val) / length * (input_range[1] - input_range[0]) + input_range[0]

def sideLength(x,dim):
    
    assert(dim == 1 or dim == 2 or dim == 3)
    l = int(round((x ** (1.0 / dim))))
    assert(x == l ** dim)
    return l

def reShape(x,dim):
    
    if (dim == 1):
        return x
    
    l = sideLength(x.size,dim)
    
    if dim == 2:
        return np.reshape(x,(l,l))
    else: 
        return np.reshape(x,(l,l,l))

def smooth(x,window_ratio=50,window='hanning'):
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
    assert(window_ratio > 1)
    
    assert(x.ndim == 1 or x.ndim == 2 or x.ndim == 3) 
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
      
    if x.ndim == 1:
        return smoothWork(x, len(x)/window_ratio)
    elif x.ndim == 2:
        window_len = len(x[0])/window_ratio
        for i in range(len(x)):
            x[i] = smoothWork(x[i], window_len)
        return x
    else:
        window_len = len(x[0][0])/window_ratio
        for i in range(len(x)):
            for j in range(len(x[0])):
                x[i][j] =  smoothWork(x[i][j], window_len)
        return x
  
def smoothWork(x,window_len=100,window='hanning'):
  
    if window_len<3:
        return x
    
    l = len(x)
    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y = np.convolve(w/w.sum(),s,mode='valid')
    y = y[(window_len/2-1):(window_len/2-1)+l]
    assert(len(y) == l)
    return y