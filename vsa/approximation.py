import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
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

    verbose_learn = False
    verbose_probe = False

    def __init__(self, fn=None, size=1000):
        self.fn = fn
        self.T = None
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
                        idx = j * len(A_x) + i
                        B_i = self.fn(A_x_i, A_y_i)  # evaluate ith sample
                        HRR_A = HRR((A_x_i, A_y_i), valid_range=input_range)
                        HRR_B = HRR(B_i, valid_range=output_range)
                        samples[idx] = (HRR_B % HRR_A).memory  # probe HRR
                        if Approximation.verbose_learn:
                            print("learning f({}, {}) = {}".format(A_x_i, A_y_i, B_i))
                            HRR_A.plot(HRR_A.reverse_permute(HRR_A.memory))
                            HRR_B.plot(HRR_B.reverse_permute(HRR_B.memory))
                            temp_B = HRR_A * HRR('', memory=samples[idx])
                            print("probed sample {}:".format(idx))
                            temp_B.plot(temp_B.reverse_permute(temp_B.memory))
                            print("sample mem:")
                            HRR_B.plot(HRR_B.reverse_permute(samples[idx]))
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
            if Approximation.verbose_probe:
                print("A * T = B")
                A.plot(A.reverse_permute(A.memory))
                B.plot(B.reverse_permute(B.memory))
            temp = B.decode(return_list=True, decode_range=output_range)
            if len(temp) > 1:
                Y_hrr[i] = temp[1][0]
                Y_hrr2[i] = temp[0][0]
            elif len(temp) > 0:
                Y_hrr[i] = temp[0][0]
                Y_hrr2[i] = temp[0][0]
            else:
                Y_hrr[i] = np.nan
                Y_hrr2[i] = np.nan
            if len(temp) > 1:
                temp = B.decode(return_list=False, suppress_value=x, decode_range=output_range)
                #print("suppress_value: {}".format(temp))
                Y_hrrsupp[i] = temp
            else:
                Y_hrrsupp[i] = np.nan
            #Y_hrr[i] = temp
            Y_np[i] = self.fn(x)
            #print("HRR: f({}) = 2nd({}) 1st({}) suppr({}) / truth: {} error: {}".format(x, Y_hrr[i], Y_hrr2[i], Y_hrrsupp[i], Y_np[i], Y_hrrsupp[i] - Y_np[i]))
        plt.figure()
        h_np, = plt.plot(X, Y_np, 'g', label="Ground truth")
        h_hrr, = plt.plot(X, Y_hrr, 'cx--', label="2nd peak if avail")
        h_hrr2, = plt.plot(X, Y_hrr2, 'bx--', label="1st peak")
        h_suppr, = plt.plot(X, Y_hrrsupp, 'rx-', label="Suppressed input x")
        plt_handles = [h_np, h_hrr, h_hrr2, h_suppr]
        plt.legend(handles=plt_handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.show()

    def plot3d_result(self, input_range, output_range, n_samples=(10, 10)):
        # only 2d input is supported
        assert(len(input_range) == 2)
        assert(isinstance(input_range[0], (frozenset, list, np.ndarray, set, tuple)))
        assert(len(input_range[0]) == 2)
        # only scalar output is supported
        assert(len(output_range) == 2)
        assert(isinstance(output_range[0], float) or isinstance(output_range[0], numbers.Integral))

        X = np.linspace(input_range[0][0], input_range[0][1], n_samples[0])
        Y = np.linspace(input_range[1][0], input_range[1][1], n_samples[1])
        X, Y = np.meshgrid(X, Y)

        # reserve memory for Z values
        Z_hrr, Z_np = np.meshgrid(np.empty(n_samples[0], dtype=float), np.empty(n_samples[1], dtype=float))
        Z_hrr2, Z_hrrsupp = np.meshgrid(np.empty(n_samples[0], dtype=float), np.empty(n_samples[1], dtype=float))

        for i, row in enumerate(X):
            for j, cell in enumerate(row):
                A = HRR((X[i][j], Y[i][j]), valid_range=input_range)
                B = A * self.T
                #A.plot(A.reverse_permute(A.memory))
                #B.plot(B.reverse_permute(B.memory))
                temp = B.decode(return_list=True, decode_range=output_range)
                if len(temp) > 1:
                    Z_hrr[i][j] = temp[1][0]
                    Z_hrr2[i][j] = temp[0][0]
                elif len(temp) > 0:
                    Z_hrr[i][j] = temp[0][0]
                    Z_hrr2[i][j] = temp[0][0]
                else:
                    Z_hrr[i][j] = np.nan
                    Z_hrr2[i][j] = np.nan
                if len(temp) > 1:
                    #temp = B.decode(return_list=False, suppress_value=x, decode_range=output_range)
                    #print("suppress_value: {}".format(temp))
                    #Z_hrrsupp[i] = temp
                    pass
                else:
                    Z_hrrsupp[i] = np.nan
                Z_np[i][j] = self.fn(X[i][j], Y[i][j])
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        surf = ax.plot_surface(X, Y, Z_hrr, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_zlim(output_range[0] - 0.01, output_range[1] + 0.01)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def verify(self, input_tuples, input_range, output_range):
        for tpl in input_tuples:
            truth = self.fn(*tpl)
            truth = [truth] if isinstance(truth, float) or isinstance(truth, numbers.Integral) else truth # convert to list
            truth_s = ["{:10.3f}".format(v) for v in truth] # format numbers nicely
            A = HRR(tpl, valid_range=input_range)
            B = A * self.T
            if Approximation.verbose_probe:
                print("A * T = B (expect {})".format(truth_s))
                print("A:")
                A.plot(A.reverse_permute(A.memory))
                print("T:")
                A.plot(A.reverse_permute(self.T.memory))
                print("B:")
                B.plot(B.reverse_permute(B.memory))
            val = B.decode(return_list=True, decode_range=output_range)
            # val is a list of tuples -> extract up to two values
            # at least one result:
            val1 = val[0][0] if len(val) > 0 else [np.nan]
            val1 = [val1] if isinstance(val1, float) or isinstance(val1, numbers.Integral) else val1 # convert to list
            val1_s = ["{:10.3f}".format(v) for v in val1] if len(val) > 0 else ["{:10.3f}".format(np.nan)] # format numbers nicely
            assert(len(val1) == len(truth))
            err1 = [v - t for v, t in zip(val1, truth)] # difference
            err1_s = ["{:10.3f}".format(v) for v in err1] # format numbers nicely
            # at least two results:
            val2 = val[1][0] if len(val) > 1 else [np.nan]
            val2 = [val2] if isinstance(val2, float) or isinstance(val2, numbers.Integral) else val2 # convert to list
            val2_s = ["{:10.3f}".format(v) for v in val2] if len(val) > 1 else ["{:10.3f}".format(np.nan)] # format numbers nicely
            assert(len(val2) == len(truth))
            err2 = [v - t for v, t in zip(val2, truth)] # difference
            err2_s = ["{:10.3f}".format(v) for v in err2] # format numbers nicely
            print("truth: {}  HRR: {}/{}{}{}  {}/{}{}{}".format(
                truth_s,
                val1_s,
                bcolors.WARNING if any(e > 0.01 for e in err1) or any(e < -0.01 for e in err1) else bcolors.OKGREEN,
                err1_s,
                bcolors.ENDC,
                val2_s,
                bcolors.WARNING if any(e > 0.01 for e in err2) or any(e < -0.01 for e in err2) else bcolors.OKGREEN,
                err2_s,
                bcolors.ENDC))


