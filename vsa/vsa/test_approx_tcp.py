from hrr import HRR
import matplotlib.pyplot as plt
import numpy as np
from approximation import Approximation

import sys
sys.path.append("../tcp")
from tcp import TCP

HRR.visualize = True
HRR.verbose = False

armlengths = [1.0, 1.0]

def fn(x, y):
    tcp = TCP([x, y], arms=2, armlengths=armlengths)
    tcp.computeTcp()
    return tcp.tcp # should be list of length 3

input_range = [(0.0, 360.0), (0.0, 360.0)]
sr = np.sum(armlengths)
output_range = [(-sr, sr), (-sr, sr)]

appr = Approximation(fn=fn, size=3000)
appr.learn(input_range=input_range, output_range=output_range, n_samples=(20, 20), stddev=0.02, use_incremental=False)

input_tuples = []
S = np.linspace(0.0, 360.0, 10)
for joint1 in S:
    for joint2 in S:
        input_tuples.append((joint1, joint2))
appr.verify(input_tuples=input_tuples, input_range=input_range, output_range=output_range)
