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