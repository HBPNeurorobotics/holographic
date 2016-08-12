import numpy as np
import matplotlib.pyplot as plt
import numbers
from hrr import HRR

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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

    def learn(self, input_range, output_range, n_samples=200, fn=None, stddev=0.03, use_incremental=True):
        if fn is not None:
            self.fn = fn
            HRR.reset_kernel()
        #HRR.input_range = np.array([in_range[0], in_range[1]])
        HRR.stddev = stddev
        #if isinstance(n_samples, float) or isinstance(n_samples, numbers.Integral):
        #    n_samples = np.tuple(n_samples)
        #if isinstance(input_range[0], float) or isinstance(input_range[0], numbers.Integral):
        #    input_range[0] = np.tuple(input_range[0])
        #if isinstance(input_range[1], float) or isinstance(input_range[1], numbers.Integral):
        #    input_range[1] = np.tuple(input_range[1])
        #if isinstance(output_range[0], float) or isinstance(output_range[0], numbers.Integral):
        #    output_range[0] = np.tuple(output_range[0])
        #if isinstance(output_range[1], float) or isinstance(output_range[1], numbers.Integral):
        #    output_range[1] = np.tuple(output_range[1])
        #assert(len(input_range[0]) == len(input_range[1]) == len(output_range[0]) == len(output_range[1]) == len(n_samples))

        #if len(n_samples) == 1:
        if isinstance(n_samples, float) or isinstance(n_samples, numbers.Integral):
            # 1D function
            # create n_samples evenly spaced sampling points for input space
            A = np.linspace(float(input_range[0]), float(input_range[1]), n_samples)
            if use_incremental:
                # initialize T
                B_0 = self.fn(A[0])
                self.T = HRR(B_0, valid_range=output_range) % HRR(A[0], valid_range=input_range)
                for A_i in A[1:]:
                    B = self.fn(A_i)
                    self.T = self.T ** (HRR(B, valid_range=output_range) % HRR(A, valid_range=input_range)) # update T
            else:
                samples = np.empty((n_samples, HRR.size), dtype=float)
                for i, A_i in enumerate(A):
                    B_i = self.fn(A_i)  # evaluate ith sample
                    HRR_A = HRR(A_i, valid_range=input_range)
                    HRR_B = HRR(B_i, valid_range=output_range)
                    samples[i] = (HRR_B % HRR_A).memory  # probe HRR
                    #HRR_A.plot(HRR_A.reverse_permute(HRR_A.memory))
                    #HRR_B.plot(HRR_B.reverse_permute(HRR_B.memory))
                    #HRR_B.plot(HRR_B.reverse_permute(samples[i]))
                self.T = HRR(0, generator=samples)
        elif len(n_samples) == 2:
            # 2D function
            A_x = np.linspace(float(input_range[0][0]), float(input_range[0][1]), n_samples[0]) # samples for X-Axis
            A_y = np.linspace(float(input_range[1][0]), float(input_range[1][1]), n_samples[1]) # samples for Y-axis
            if use_incremental:
                # initialize T
                B_0 = self.fn(A_x[0], A_y[0])
                self.T = HRR(B_0, valid_range=output_range[0]) % HRR((A_x[0], A_y[0]), valid_range=input_range[0])
                for A_x_i in A_x[1:]: # iterate over X
                    for A_y_i in A_y[1:]: # iterate over Y
                        B = self.fn(A_x_i, A_y_i)
                        self.T = self.T ** (HRR(B, valid_range=output_range) % HRR((A_x_i, A_y_i), valid_range=input_range)) # update T
            else:
                samples = np.empty((n_samples[0] * n_samples[1], HRR.size), dtype=float)
                for i, A_x_i in enumerate(A_x):
                    for j, A_y_i in enumerate(A_y):
                        B_i = self.fn(A_x_i, A_y_i)  # evaluate ith sample
                        HRR_A = HRR((A_x_i, A_y_i), valid_range=input_range)
                        HRR_B = HRR(B_i, valid_range=output_range)
                        samples[i] = (HRR_B % HRR_A).memory  # probe HRR
                        #print("learning f({}, {}) = {}".format(A_x_i, A_y_i, B_i))
                        #HRR_A.plot(HRR_A.reverse_permute(HRR_A.memory))
                        #HRR_B.plot(HRR_B.reverse_permute(HRR_B.memory))
                        #HRR_B.plot(HRR_B.reverse_permute(samples[i]))
                self.T = HRR(0, generator=samples)
        else:
            raise ValueError("Dimensions > 2 not implemented yet")

        #self.T.plot()
        #print("learn: {}".format(self.T.memory))

    def plot_result(self, input_range, output_range, n_samples=10):
        X = np.linspace(input_range[0], input_range[1], n_samples)
        Y_hrr = np.empty(n_samples, dtype=float)
        Y_hrr2 = np.empty(n_samples, dtype=float)
        Y_hrrsupp = np.empty(n_samples, dtype=float)
        Y_np = np.empty(n_samples, dtype=float)
        for i, x in enumerate(X):
            A = HRR(x, valid_range=input_range)
            B = A * self.T
            #A.plot(A.reverse_permute(A.memory))
            #B.plot(B.reverse_permute(B.memory))
            temp = B.decode(return_dict=True, decode_range=output_range)
            if len(temp) > 1:
                Y_hrr[i] = temp.keys()[1]
                Y_hrr2[i] = temp.keys()[0]
            elif len(temp) > 0:
                Y_hrr[i] = temp.keys()[0]
                Y_hrr2[i] = temp.keys()[0]
            else:
                Y_hrr[i] = np.nan
                Y_hrr2[i] = np.nan
            if len(temp) > 1:
                temp = B.decode(return_dict=False, suppress_value=x, decode_range=output_range)
                #print("suppress_value: {}".format(temp))
                Y_hrrsupp[i] = temp
            else:
                Y_hrrsupp[i] = np.nan
            #Y_hrr[i] = temp
            Y_np[i] = self.fn(x)
            print("HRR: f({}) = 2nd({}) 1st({}) suppr({}) / truth: {} error: {}".format(x, Y_hrr[i], Y_hrr2[i], Y_hrrsupp[i], Y_np[i], Y_hrrsupp[i] - Y_np[i]))
        plt.figure()
        h_np, = plt.plot(X, Y_np, 'g', label="Ground truth")
        h_hrr, = plt.plot(X, Y_hrr, 'cx--', label="2nd peak if avail")
        h_hrr2, = plt.plot(X, Y_hrr2, 'bx--', label="1st peak")
        h_suppr, = plt.plot(X, Y_hrrsupp, 'rx-', label="Suppressed input x")
        plt_handles = [h_np, h_hrr, h_hrr2, h_suppr]
        plt.legend(handles=plt_handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.show()

    def verify(self, input_tuples, input_range, output_range):
        for tpl in input_tuples:
            truth = self.fn(*tpl)
            A = HRR(tpl, valid_range=input_range)
            B = A * self.T
            val = B.decode(return_dict=True, decode_range=output_range)
            #print("f({}) = {} / truth: {} diff: ".format(tpl, val, truth))
            val1 = val.keys()[0] if len(val) > 0 else 0.0
            err1 = val1 - truth
            val2 = val.keys()[1] if len(val) > 1 else 0.0
            err2 = val2 - truth
            print("{}/{}{}{} {}/{}{}{}".format(
                val1,
                bcolors.WARNING if err1 > 0.01 or err1 < -0.01 else bcolors.OKGREEN,
                err1,
                bcolors.ENDC,
                val2,
                bcolors.WARNING if err2 > 0.01 or err2 < -0.01 else bcolors.OKGREEN,
                err2,
                bcolors.ENDC))


