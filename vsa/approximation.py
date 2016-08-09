import numpy as np
import matplotlib.pyplot as plt
from hrr import HRR

class Approximation:

    fn = None
    T = None

    def __init__(self, fn=None, size=1000):
        self.fn = fn
        HRR.reset_kernel()
        if size is not None:
            HRR.set_size(size)
        else:
            HRR.set_size(HRR.size)

    def learn(self, in_range=(-1.0, 1.0), n_samples=200, fn=None, stddev=0.03):
        if fn is not None:
            self.fn = fn
            HRR.reset_kernel()
        HRR.input_range = np.array([in_range[0], in_range[1]])
        HRR.stddev = stddev
        # create n_samples evenly spaced sampling points for input space
        #print("HRR size: {}, input_range: {}, stddev {}".format(HRR.size, HRR.input_range, HRR.stddev))
        A = np.linspace(float(in_range[0]), float(in_range[1]), n_samples)
        #print("A: {}".format(A))
        samples = np.empty((n_samples, HRR.size), dtype=float)
        # sample function and store one HRR per sample
        for i, A_i in enumerate(A):
            B_i = self.fn(A_i)  # evaluate ith sample
            samples[i] = (HRR(B_i) % A_i).memory  # probe HRR

        #print("samples: {}".format(samples))
        self.T = HRR(0, generator=samples)
        #print("learn: {}".format(self.T))

    def plot_result(self, n_samples=10):
        X = np.linspace(HRR.input_range[0], HRR.input_range[1], n_samples)
        Y_hrr = np.empty(n_samples, dtype=float)
        Y_np = np.empty(n_samples, dtype=float)
        for i, x in enumerate(X):
            A = HRR(x)
            B = A * self.T
            #HRR.plot(HRR.reverse_permute(HRR(x).memory))
            #HRR.plot(HRR.reverse_permute((A * self.T).memory))
            Y_hrr[i] = B.decode()
            Y_np[i] = self.fn(x)
            #print("HRR: f({}) = {} / truth: {} error: {}".format(x, Y_hrr[i], Y_np[i], Y_np[i] - Y_hrr[i]))
        plt.figure()
        plt.plot(X, Y_np, 'g')
        plt.plot(X, Y_hrr, 'rx-')
        plt.show()

