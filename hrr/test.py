from hrr import *
#import core
import numpy as np
import sys

import matplotlib.pyplot as plt

import cv2

#a = HRR(5) * 6
#b = HRR(1) * 2
#k = HRR(4) * 1

#c = a + b + k

#print(c/1)

#sys.exit()

# define list of items to remember
HRR.reset_kernel()
HRR.set_default_size(100 * 100 * 3)
HRR.input_range = [0, 10]
HRR.stddev = 0.001

# load images from files
dataset = {}
dataset[1] = 'circle-red.png'
dataset[9] = 'circle-green.png'

m = None
for k in dataset:
    img = cv2.imread(dataset[k])
    img = cv2.resize(img, (100,100))
    img_data = np.array(img).reshape(img.shape[0] * img.shape[1] * img.shape[2])
    print(max(img_data))
    img_data /= float(max(img_data))
    #img_data -= 0.5

    print(max(img_data))

    symbol = HRR(k, memory=img_data) * k
    if m is not None:
        m += symbol
    else:
        m = symbol

#features = [ "door", "tree", "car", "table", "computer" ]
#values = [ (10,20), (90,46), (22,31), (50,1), (1,50) ]
#m = None
#for i in range(len(features)):
#    if m is None:
#        m = HRR(features[i]) * values[i]
#    else:
#        new_item = HRR(features[i]) * values[i]
#        m += new_item

#print("Items memorized.")
#print(m.memory)

cam = cv2.VideoCapture(0)

#plt.ion()
#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)
while True:
    ret, img = cam.read(0)
    
    rect = ((100,100),(300,300))
    cv2.rectangle(img, rect[0], rect[1], (0,255,0), 3)
    
    img_crop = img[rect[0][0]:rect[1][0], rect[0][1]:rect[1][1]]
    
    img_probe = cv2.resize(img_crop, (100,100))
    img_probe = np.array(img_probe).reshape(img_probe.shape[0] * img_probe.shape[1] * img_probe.shape[2])
    
    #print('before {} {}'.format(np.max(img_probe), np.min(img_probe)))
    img_probe /= float(max(img_probe))
    #img_probe -= 0.5
    #print('after {} {}'.format(np.max(img_probe), np.min(img_probe)))
    
    res = m % HRR('-', memory=img_probe)
    out = np.argmax(res.memory) / float(HRR.default_size) * 10.0
    
    print(res.memory)
    
    cv2.imshow("image", img)
    cv2.imshow("crop", img_crop)
    cv2.waitKey(30)

