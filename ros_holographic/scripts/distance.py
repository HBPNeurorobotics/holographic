#!/usr/bin/env python

import sys
import pickle
import numpy as np
from numpy.linalg import norm


def distance(file_one, file_other):
    one = pickle.load(open(file_one, 'rb'))
    other = pickle.load(open(file_other, 'rb'))

    scale = norm(one)*norm(other)

    if scale==0:
        return 0
    else:
        return np.dot(one, other)/(scale)

args = sys.argv
print distance(args[1], args[2])
