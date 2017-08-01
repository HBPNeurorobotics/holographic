import numpy as np
import matplotlib.pyplot as plt

def getLegend(v,start):
    sh = len(v)
    linestyles = ['-', '--', '-.', ':']
    for i in range(sh):
        plt.plot(v[i], linestyles[3-i*4/sh], label=start*pow(4,i))
    plt.axis([0,len(v[0]),0,1.01])
    plt.xlabel('Bindings')
    plt.ylabel('Accuracy')
    legend = plt.legend(loc='lower left', shadow=True, fontsize='x-large')
    plt.show()
