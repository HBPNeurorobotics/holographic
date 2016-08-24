import math
import numpy as np
import operator
from vsa import VSA

class CHRR(VSA):
    
    ## The constructor.
    #
    #  @param self The object pointer.
    #  @param input_value The value for which the CHRR is constructed. May be of different types.
    #  @param memory A float vector which is the symbol by which the input value will be represented.
    def __init__(self, input_value, memory=None):
       
        self.label = input_value

        if memory is not None:
            self.memory = memory 
        else:
            self.memory = self.encode(input_value)  
            
    ## Overload of the "+" operand.
    #
    #  Creates a new HRR by transforming the angles into vectors, adding these, and transforming them back.
    #
    #  @param self The object pointer.
    #  @param operand The second operand of the operation.
    #  @return The new HRR.
    def __add__(self, operand):
        
        if operand.__class__ != self.__class__:
            operand = CHRR(operand)
            
        memory = np.empty(self.size, dtype=float)    
        for i in range(self.size):
            memory[i] = self.vectorAngleCalculator(self.memory[i],operand.memory[i])
        return CHRR('-', memory=memory)

    ## Overload of the "-" operand.
    #
    #  Creates a new HRR by transforming the angles into vectors, subtracting these, and transforming them back.
    #
    #  @param self The object pointer.
    #  @param operand The second operand of the operation.
    #  @return The new HRR.
    def __sub__(self, operand):
        
        if operand.__class__ != self.__class__:
            operand = CHRR(operand)
            
        memory = np.empty(self.size, dtype=float)    
        for i in range(self.size):
            memory[i] = self.vectorAngleCalculator(self.memory[i],operand.memory[i],True)
        return CHRR('-', memory=memory)   
    
    def vectorAngleCalculator(self, angle1,angle2,sub=False):
        if sub:
            factor = -1
        else: 
            factor = 1
        
        return np.math.atan2(np.math.sin(angle1) + factor * np.math.sin(angle1), np.math.cos(angle1) + factor * np.math.cos(angle1))
    
    ## Overload of the "*" operand.
    #
    #  Creates a new HRR containing by binding the other two.
    #
    #  @param self The object pointer.
    #  @param operand The second operand of the operation.
    #  @return The new HRR.        
    def __mul__(self, operand):
        if operand.__class__ != self.__class__:
            operand = CHRR(operand)
        memory = self.bind(self.memory, operand.memory) #perform binding       
        
        return CHRR('-', memory=memory)
    
    ## Overload of the "%" operand.
    #
    #  Creates a new HRR containing by probing the first with the second.
    #
    #  @param self The object pointer.
    #  @param operand The second operand of the operation.
    #  @return The new HRR.    
    def __mod__(self, operand):
        
        if operand.__class__ != self.__class__:
            operand = CHRR(operand)
        memory = self.probe(self.memory, operand.memory) #perform unbinding
        
        return CHRR('-', memory=memory)
    
    ## Overload of the "/" operand.
    #
    #  Probes the first HRR with the second and decodes the result, returning it as a dictionary.
    #  The value of the keys represents the distance to the actual mapping.
    #
    #  @param self The object pointer.
    #  @param operand The second operand of the operation.
    #  @return A dictionary of the decoded values. Can also be empty or contain more than one value.       
    def __div__(self, operand):
    
        if operand.__class__ != self.__class__:
            operand = CHRR(operand)        
        memory = self.probe(self.memory, operand.memory) #perform unbinding
        
        if self.visualize:
            print("Output:")
            self.plot(memory)
        
        return self.decode(memory)    
    
    ## Generates a vector of random angles.
    #
    #  @param self The object pointer.
    #  @return The generated vector.           
    def generateVector(self):
        return np.array([np.random.uniform(-math.pi,math.pi) for i in range(self.size)])  
    
    ## Binds two vectors by adding them together keeping the modulo of the angle addition for each entry.
    #
    #  @param self The object pointer.
    #  @param one The first of the two vectors.
    #  @param one The second vector.
    #  @return The result of binding the two vectors.    
    def bind(self, one, other):
        result = one + other
        return result % (2*math.pi)   
    
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