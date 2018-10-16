import numpy as np
import scipy as sp
import scipy.ndimage
from scipy.signal import argrelextrema
from numpy.fft import fft, ifft
from numpy.linalg import norm
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
    peak_min = 0                    # Absolute value used to detect Gaussian peaks
    peak_min_ratio =  10            # Minimum factor by which a gaussian peak needs be larger than the average of noise.
    window_ratio = 50                   # Ratio by which window length for smoothing is divided

    incremental_weight = 0.1        # Weight used for induction.

    ## Overload of the "+" operand.
    #
    #  Creates a new HRR containing the sum of the other two memories as its own.
    #
    #  @param self The object pointer.
    #  @param operand The second operand of the operation.
    #  @return The new HRR.
    def __add__(self, operand):

        if operand.__class__ != self.__class__:
            operand = type(self)(operand)
        return type(self)('-', memory=self.memory + operand.memory)

    ## Overload of the "-" operand.
    #
    #  Creates a new HRR containing the subtraction of the other two memories as its own.
    #
    #  @param self The object pointer.
    #  @param operand The second operand of the operation.
    #  @return The new HRR.
    def __sub__(self, operand):

        if operand.__class__ != self.__class__:
            operand = type(self)(operand)
        return type(self)('-', memory=self.memory - operand.memory)

    ## Overload of the "*" operand.
    #
    #  Creates a new HRR containing by binding the other two with the help of circular convolution.
    #
    #  @param self The object pointer.
    #  @param operand The second operand of the operation.
    #  @return The new HRR.
    def __mul__(self, operand):
        if operand.__class__ != self.__class__:
            operand = type(self)(operand)
        memory = self.bind(self.memory, operand.memory) #perform binding

        return type(self)('-', memory=memory)

    ## Overload of the "%" operand.
    #
    #  Creates a new HRR containing by probing the first with the second via periodic correlation.
    #
    #  @param self The object pointer.
    #  @param operand The second operand of the operation.
    #  @return The new HRR.
    def __mod__(self, operand):

        if operand.__class__ != self.__class__:
            operand = type(self)(operand)
        memory = self.probe(self.memory, operand.memory) #perform unbinding

        return type(self)('-', memory=memory)

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
            operand = type(self)(operand)
        memory = self.probe(self.memory, operand.memory) #perform unbinding

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
            operand = type(self)(operand)

        return type(self)('-', memory = self.memory - self.incremental_weight * (self.memory - operand.memory))

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

    def decodeCoordinate(self, memory=None, dim=1, return_list=False, suppress_value=None, decode_range=None):

        assert(dim == 1 or dim == 2 or dim == 3)

        if memory is None:
            memory = self.memory

        memory = helpers.normalize(memory)

        if decode_range is None:
            decode_range = self.valid_range
        if decode_range is None:
            raise ValueError("Decoding scalar values requires valid range (valid_range or decode_range parameter)")



        assert(len(decode_range) == dim)

        memory = self.reverse_permute(memory)

        if self.visualize:
            print("Output Reverse:")
            self.plot(np.reshape(memory,self.size))

        memory = helpers.smooth(helpers.reShape(memory, dim),self.window_ratio)
        l = helpers.sideLength(memory.size, dim)

        if self.visualize:
            print("Output Smooth pre:")
            self.plot(np.reshape(memory,self.size))

        if suppress_value is not None:
            memory = self.deductValue(memory,supress_value,HRR.valid_range)
            if self.visualize:
                print("Output Smooth (after suppression):")
                self.plot(np.reshape(memory,self.size))

        result = []

        if(self.peak_min == 0):
            self.peak_min = np.max(memory)/2

        while np.max(memory) > self.peak_min_ratio * abs(np.mean(memory)) + self.peak_min:

            spot = list(np.unravel_index(np.argmax(memory),memory.shape))

            for i in range(dim):
                spot[i] = helpers.reverse_scale(spot[i], l, decode_range[i])

            result.append((spot, 1))
            if return_list is False:
                return spot
            memory = self.deductValue(memory,spot,HRR.valid_range,dim, np.max(memory))
            if self.visualize:
                print("Output Post Deduction:")
                self.plot(np.reshape(memory,self.size))

        if len(result) == 0 and suppress_value is not None:
            return [(np.nan, 1)] if return_list else np.nan
        return result

    def deductValue(self, memory, value, input_range, dim = 1, height = 1):

        #result = self.permute(helpers.normalize(self.coordinate_encoder(input_value, encode_range)))
        assert(input_range is not None)
        # suppress the given values in the memory
        if not isinstance(value, (frozenset, list, np.ndarray, set, tuple)):
            value = [value]
        compensate = self.coordinate_encoder(value, input_range)
        # we have to smooth this to ensure correct alignment with the current memory
        compensate = helpers.reShape(compensate,dim)
        compensate = helpers.smooth(compensate, self.window_ratio)
        compensate[:] = [x * -height for x in compensate]

        if self.visualize:
            print("Output Supressed Value:")
            self.plot(np.reshape(compensate,self.size))


        memory += compensate
        return memory
        #for i, v in enumerate(suppress_value):
        #    assert(len(suppress_value) == len(input_range))
        #    compensate = self.scalar_encoder(v, len(memory), input_range[i])
        #    compensate[:] = [x * -abs(np.max(memory)) for x in compensate]
        #    memory += compensate

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
                result = self.permute(helpers.normalize(self.scalar_encoder(input_value, self.size, encode_range[0])))
                if self.visualize:
                    print("Encoded ", input_value)
                    self.plot(result)
            elif isinstance(input_value, (frozenset, list, np.ndarray, set, tuple)):
                result = self.permute(helpers.normalize(self.coordinate_encoder(input_value, encode_range)))
                if self.visualize:
                    print("Encoded ", input_value)
                    self.plot(result)
            else:
                result = VSA.encode(self, input_value)
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
        assert(self.size == int(round((self.size ** (1.0 / nr)))) ** nr)

        ## compute individual lengths
        #length = int(HRR.size / nr)
        #length_last = HRR.size - length * (nr - 1)
        #out = []
        #for i in range(nr-1):
        #    enc = self.scalar_encoder(coordinates[i], length, encode_range[i])
        #    out.extend(enc)
        ## encode the last segment
        #enc = self.scalar_encoder(coordinates[nr-1], length_last, encode_range[nr-1])
        #out.extend(enc)

        vlen = helpers.sideLength(self.size,nr)

        out = self.scalar_encoder(coordinates[0], vlen, encode_range[0])
        for i in range(nr-1):
            out = np.kron(out, self.scalar_encoder(coordinates[i+1], vlen, encode_range[i+1]))
        if self.visualize:
            print("Encoded Coordinate")
            self.plot(out)

        assert(len(out) == self.size)

        return out

    ## Performs periodic correlation on two symbolic vectors.
    #
    #  @param self The object pointer.
    #  @param one The first of the two vectors.
    #  @param one The second vector.
    #  @return The result of periodic correlation as a vector.
    def probe(self, one, other):
        return helpers.normalize(ifft(fft(one) * fft(other).conj()).real)

    ## Performs circular convolution on two symbolic vectors.
    #
    #  @param self The object pointer.
    #  @param one The first of the two vectors.
    #  @param one The second vector.
    #  @return The result of circular convolution as a vector.
    def bind(self, one, other):
        return helpers.normalize(np.real(ifft(fft(one)*fft(other))))

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
        return np.dot(one,other)/(scale)

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
