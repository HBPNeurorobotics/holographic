from hrr import HRR, smooth
import matplotlib.pyplot as plt
import numpy as np
import cv2

import os, shutil
folder = '/Users/peric/dev/hrr/results'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)
       
HRR.stddev = 0.01
HRR.input_range = [0, 10]
HRR.default_size = 10000

labels = [2, 8]

path = []
path.append([ (3,4), (4,4), (5,4), (6,4), (6,3), (6,2), (6,1), (7,1), (8,1), (8,2)])
path.append([ (0,0), (1,1), (2,2), (3,3), (4,4), (5,5), (6,6)])
#items = ["computer", "car", "table", "chair", "door", "book"]

m = None
m_time = None
xx, yy = [], []
for p in range(len(labels)):
    print("Processing path {}...".format(p))
    for k in range(len(path[p])):
        print("Index {}".format(k))
        path_representation = HRR(path[p][k]) * labels[p]
        time_representation = HRR(path[p][k]) * labels[p] * k
        
        xx.extend([path[p][k][0]])
        yy.extend([10-path[p][k][1]])
        
        if m is None:
            m = path_representation
        else:
            m += path_representation
        
        if m_time is None:
            m_time = time_representation
        else:
            m_time += time_representation
            

# get the intersection coordinates
A = labels[0]
B = labels[1]
int_A = m % HRR(A)
int_A.memory = int_A.reverse_permute(int_A.memory)
int_B = m % HRR(B)
int_B.memory = int_B.reverse_permute(int_B.memory)

int_C = int_A.memory * int_B.memory
int_C /= np.linalg.norm(int_C) 

#intersection_coords.memory = intersection_coords.reverse_permute(intersection_coords.memory)

plt.figure()
xx = range(len(int_C))
plt.plot(xx, int_C)
plt.show()

#intersection_coords.memory = intersection_coords.permute(intersection_coords.memory)

# get time for path A
time_A = m_time % (intersection_coords * A)
#time_A.memory = time_A.permute(time_A.memory)
time_B = m_time % (intersection_coords * B)

plt.figure()
#time_A.memory = time_A.permute(time_A.memory)
#time_A.memory = smooth(time_A.memory, window_len=50)
xx = range(len(time_B.memory))
plt.plot(xx, time_B.memory)
plt.show()


        