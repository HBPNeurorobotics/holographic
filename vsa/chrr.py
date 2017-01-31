import math
import numpy as np
import operator
import random
from hrr import HRR

class CHRR(HRR):
    
    distance_threshold = 0.9       # Distance from which similarity is accepted.
    
    ## Overload of the "+" operand.
    #
    #  Creates a new HRR by transforming the angles into vectors, adding these, and transforming them back.
    #
    #  @param self The object pointer.
    #  @param operand The second operand of the operation.
    #  @return The new HRR.
    def __add__(self, operand):
        
        if operand.__class__ != self.__class__:
            operand = type(self)(operand)
            
        memory = np.empty(self.size, dtype=float)    
        for i in range(self.size):
            memory[i] = self.vectorAngleCalculator(self.memory[i],operand.memory[i])
        return type(self)('-', memory=memory)

    ## Overload of the "-" operand.
    #
    #  Creates a new HRR by transforming the angles into vectors, subtracting these, and transforming them back.
    #
    #  @param self The object pointer.
    #  @param operand The second operand of the operation.
    #  @return The new HRR.
    def __sub__(self, operand):
        
        if operand.__class__ != self.__class__:
            operand = type(self)(operand)
            
        memory = np.empty(self.size, dtype=float)    
        for i in range(self.size):
            memory[i] = self.vectorAngleCalculator(self.memory[i],operand.memory[i],True)
        return type(self)('-', memory=memory)   
    
    def vectorAngleCalculator(self, angle1,angle2,sub=False):
        if sub:
            factor = -1
        else: 
            factor = 1
      
        return np.math.atan2(np.math.sin(angle1) + factor * np.math.sin(angle2), np.math.cos(angle1) + factor * np.math.cos(angle2))
        
    ## Generates a normalized vector of random values.
    #
    #  @param self The object pointer.
    #  @return The generated vector.  
    def generateVector(self):
        return np.array([random.gauss(-math.pi, math.pi) for i in range(self.size)])
    
    ## Binds two vectors by adding them together keeping the modulo of the angle addition for each entry.
    #
    #  @param self The object pointer.
    #  @param one The first of the two vectors.
    #  @param one The second vector.
    #  @return The result of binding the two vectors.    
    def bind(self, one, other):
        return (one + other) % (2 * math.pi) 
    
    ## Probes a vector by another one by binding it with the inverse of the second.
    #
    #  @param self The object pointer.
    #  @param one The first of the two vectors.
    #  @param one The second vector.
    #  @return The result of probing the two vectors.    
    def probe(self, one, other):
        return self.bind(one, self.inverse(other))

    ## Inverts a symbolic vector.
    #
    #  @param self The object pointer.
    #  @param vect The vector to be inverted.
    #  @return The inverted vector.  
    def inverse(self, vect):
        return ((vect * (-1)) % (2*math.pi))
    
    ## Calculates the distance between two symbolic vectors.
    #
    #  @param self The object pointer.
    #  @param one The first of the two vectors.
    #  @param one The second vector.
    #  @return The distance between the two vectors.    
    def distance(self,one, other):
        return (sum(np.cos(one - other))) / max(self.size, 1)