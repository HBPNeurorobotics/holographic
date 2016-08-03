from hrr import HRR
import matplotlib.pyplot as plt

import numpy as np

HRR.input_range = [0, 100]

n = 50 # number of items stored

all_rules = [0.0] * HRR.default_size
w = 0.3

rule = None
for i in range(n):
    A = HRR(i)
    B = HRR(i+10)
    T = A % B
    T.memory = T.permute(T.memory)
    if rule is None:
        rule = T
    else:
        rule -= T
    all_rules = all_rules - w * (all_rules - T.memory)

#all_rules /= float(n)
hrr_rule = HRR('-', memory=all_rules)

#hrr_rule.memory = hrr_rule.reverse_permute(hrr_rule.memory)
plt.figure()
xx = range(len(hrr_rule.memory))
plt.plot(xx, hrr_rule.memory)
plt.show()

cue = HRR(10)
inference = cue * hrr_rule

inference.memory = inference.reverse_permute(inference.memory)

plt.figure()
xx = range(len(inference.memory))
plt.plot(xx, inference.memory)
plt.show()