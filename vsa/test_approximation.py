from hrr import HRR
import matplotlib.pyplot as plt
import numpy as np
from approximation import Approximation
#%matplotlib inline

def fn_square(x):
    return x*x

def fn(x):
    return fn_square(x)

HRR.visualize = False
HRR.verbose = False

input_range = np.array([-3.5, 3.5])
output_range = np.array([0.0, fn_square(input_range[0])])

print("input_range: {} output_range: {}".format(input_range, output_range))

appr = Approximation(fn=fn, size=2000)
HRR.incremental_weight = 0.5
appr.learn(input_range=input_range, output_range=output_range, n_samples=200, stddev=0.02, use_incremental=False)
appr.plot_result(input_range=input_range, output_range=output_range, n_samples=19)

