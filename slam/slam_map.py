import numpy as np

class SLAM_MAP:
    
    verbose = False # Console verbose output.
    
    def __init__ (self, length=10, sight=2):
        assert(length > 0 and sight > 0 and sight < length)
        self.smap = np.zeros((length,length))
        self.length = length
        self.sight = sight
        self.mappings = {}
        
    def evalCoord(self,coord):    
        assert(len(coord) == 2 and all(c >= 0 for c in coord) and all(c < self.length for c in coord))
    
    def delCoord(self,coord):
        self.evalCoord(coord)
        self.setObject(coord)   
        
    def delObject(self,value):
        if not value in self.mappings.values():
            print("Nothing here!")
        else:
            for k,v in self.mappings.items():
                if self.mappings[k] == value:
                    self.delCoord(k)
    
    def setObject(self, coord, value=0):
        self.evalCoord(coord)
        if value == 0:
            self.mappings.pop(coord, None)
            self.smap[coord[0]][coord[1]] = 0
        else:
            self.mappings[coord] = value
            self.smap[coord[0]][coord[1]] = 1
            
    def setBot(self,coord):
        self.evalCoord(coord)
        if self.smap[coord[0]][coord[1]] != 0:
            print("Space is taken!")
        else:
            self.botPos = coord
            self.smap[coord[0]][coord[1]] = 7
         
    def detect(self):
        c = self.botPos
        x1 = c[0] - self.sight
        x2 = c[0] + self.sight
        y1 = c[1] - self.sight
        y2 = c[1] + self.sight
        self.adjustCoord(x1)
        self.adjustCoord(x2)
        self.adjustCoord(y1)
        self.adjustCoord(y2)
        
        d = {}
        
        for i in range (x1, x2):
            for j in range (y1, y2):
                if self.smap[i][j] == 1:
                    d[(i,j)] = self.mappings[(i,j)]
        
        return d
        
    def adjustCoord(self,val):
        if val < 0:
            val = 0
        elif val > self.length - 1:
            val = self.length - 1
        
    def up(self):
        if (self.botPos[0] == self.length - 1 or self.smap[self.botPos[0]+1][self.botPos[1]] != 0):
            print("Can't go up!")
        else:
            self.smap[self.botPos[0]+1][self.botPos[1]] = 7
            self.smap[self.botPos[0]][self.botPos[1]]   = 0
            self.botPos[0] += 1
        
    def down(self):
        if (self.botPos[0] == 0 or self.smap[self.botPos[0]-1][self.botPos[1]] != 0):
            print("Can't go down!")
        else:
            self.smap[self.botPos[0]-1][self.botPos[1]] = 7
            self.smap[self.botPos[0]][self.botPos[1]]   = 0
            self.botPos[0] -= 1
        
    def left(self):
        if (self.botPos[1] == 0 or self.smap[self.botPos[0]][self.botPos[1]-1] != 0):
            print("Can't go left!")
        else:
            self.smap[self.botPos[0]][self.botPos[1]-1] = 7
            self.smap[self.botPos[0]][self.botPos[1]]   = 0
            self.botPos[1] -= 1
        
    def right(self):
        if (self.botPos[1] == self.length - 1 or self.smap[self.botPos[0]][self.botPos[1]+1] != 0):
            print("Can't go right!")
        else:
            self.smap[self.botPos[0]][self.botPos[1]+1] = 7
            self.smap[self.botPos[0]][self.botPos[1]]   = 0
            self.botPos[1] += 1
        
    def showMap(self):
        print(self.smap)