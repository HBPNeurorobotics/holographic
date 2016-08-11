from hrr import HRR
import matplotlib.pyplot as plt
import numpy as np
from approximation import Approximation
#%matplotlib inline

HRR.verbose = False

def fn_square(x):
    return x*x

def fn(x):
    return fn_square(x)

HRR.visualize = False

in_range = np.array([-3.5, 3.5])
#out_range = np.array([-1.0, 1.0])
out_range = np.array([0.0, fn_square(in_range[0])])

print("in_range: {} out_range: {}".format(in_range, out_range))

appr = Approximation(fn=fn, size=2000)
HRR.incremental_weight = 0.5
appr.learn(in_range=in_range, out_range=out_range, n_samples=200, stddev=0.02, use_incremental=False)
appr.plot_result(in_range=in_range, out_range=out_range, n_samples=19)

