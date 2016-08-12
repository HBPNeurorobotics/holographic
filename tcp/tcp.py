import numpy as np

##  Tool Center Point Calculator.
# 
#   A calculator that resolves a simplified version of forward kinematics.

class TCP:
  
    verbose = False   # Console verbose output.
    
    ## The constructor.
    #
    #  This constructor can take a variety of parameters to define how the TCP will work. 
    #  The default setting only requires a list of two angles for the z-Axis of the arms in a 2D model.
    #
    #  @param self The object pointer.
    #  @param angles The list of arm angles. In 2D it needs to have the length equal to the number of arms.
    #                In 3D the list also gains an extra dimension and stores two angles for each arm,
    #                one for the x-Axis and one for the z-Axis. Angles are assumed to be in degrees.
    #  @param arms The total number of arms.
    #  @param armlengths The length of each arm, stored as a list. Must match the number of arms.
    #  @param d3 Boolean that switches to the 3D model.
    #  @param rad Boolean that switches to radian input.
    def __init__ (self, angles, arms=2, armlengths=[10,10], d3=False, rad=False):
    
        self.d3 = d3
        self.rad = rad
        assert(arms > 0)
        self.arms = arms
        assert(len(armlengths) == arms and all(a > 0 for a in armlengths))
        self.armlengths = armlengths
        self.setAngles(angles)    
        
    ## Setter for arm angles.
    #
    #  @param self The object pointer.
    #  @param angles The list of arm angles, which must have the same shape as for the constructor.
    def setAngles(self,angles):
        if not self.d3:
            assert(not isinstance(angles[0],list))
            assert(len(angles) == self.arms and all(a >= 0 for a in angles))
            self.angles = angles
        else:
            assert(len(angles) == self.arms)
            assert(all(len(a) == 2 for a in angles))
            assert(all(all(a >= 0 for a in x) for x in angles))
            self.angles = angles          
        if not self.rad:
            self.convertToRad()        
            
    ## Converter to radians.
    #
    #  Switches from degrees to radians.
    #
    #  @param self The object pointer.      
    def convertToRad(self):
        
        if not self.d3:
            for i in range(self.arms):
                self.angles[i] = np.deg2rad(self.angles[i])   
        else:
            for i in range(self.arms):
                self.angles[i][0] = np.deg2rad(self.angles[i][0]) 
                self.angles[i][1] = np.deg2rad(self.angles[i][1])    
 
    ## The TCP calculator.
    #
    #  Computes the instance of a TCP in accordance with the values used in the constructor or set afterwards.
    #
    #  @param self The object pointer. 
    def computeTcp(self):
        
        self.tcp = np.zeros(4)
        self.tcp[3] = 1
        
        for i in range(self.arms):
            
            if (self.verbose):
                print("Arm Number: ",i)
                
            trans = np.zeros((4,4))
            trans[0][0] = 1
            trans[1][1] = 1
            trans[2][2] = 1
            trans[3][3] = 1
            trans[0][3] = self.armlengths[i]
            
            if (self.verbose):
                print("Translation Matrix: ",trans)
                
            if not self.d3:
                zangle = self.angles[i]
            else:
                xangle = self.angles[i][0]
                zangle = self.angles[i][1]

            rotz = np.zeros((4,4))
            rotz[2][2] = 1
            rotz[3][3] = 1
            rotz[0][0] =  np.cos(zangle)
            rotz[0][1] = -np.sin(zangle)
            rotz[1][0] =  np.sin(zangle)
            rotz[1][1] =  np.cos(zangle)    
                
            if (self.verbose):
                print("Rotation Matrix for Z: ",rotz)
                
            if not self.d3:
                transf = np.dot(rotz,trans)
            else:
                rotx = np.zeros((4,4))
                rotx[0][0] = 1
                rotx[3][3] = 1
                rotx[1][1] =  np.cos(xangle) 
                rotx[1][2] = -np.sin(xangle) 
                rotx[2][1] =  np.sin(xangle)   
                rotx[2][2] =  np.cos(xangle) 
                
                if (self.verbose):
                    print("Rotation Matrix for X: ",rotx)
                    
                transf = np.dot(rotx, np.dot(rotz, trans))

            if (self.verbose):
                print("Transformation Matrix: ",transf)
                print("Initial TCP: ",self.tcp)
                
            self.tcp = np.dot(transf, self.tcp)
            
            if (self.verbose):
                print("Resulting TCP: ",self.tcp)
                
        if not self.d3:
            self.tcp = self.tcp[0:2]
        else:
            self.tcp = self.tcp[0:3]
                
            
