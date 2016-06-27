import numpy as np
import matplotlib.pyplot as plt

def getLegend(v,start):
    sh = len(v)
    linestyles = ['-', '--', '-.', ':']
    for i in range(sh):
        plt.plot(v[i], linestyles[3-i*4/sh], label=start*pow(2,i))
    plt.axis([0,len(v[0]),0,1.01])
    legend = plt.legend(loc='upper center', shadow=True, fontsize='x-large')
    plt.show()
