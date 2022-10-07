import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import scipy.optimize as opt
import scipy.io as sio
from scipy.interpolate import griddata
import itertools
from scipy.interpolate import interpn
import cvxpy as cvx
import pickle
# Make CBF Class
from sympy import Q


class CBF:
    def __init__(self, f, g_u, g_d, grid, dim=3, rate=10, constraints=None):
        self.data_t = {}
        self.f = f
        self.g_u = g_u
        self.g_d = g_d
        self.grid = grid
        self.dim = dim
        self.rate = rate
        self.constraints = constraints
        
    def load_data(self, prefix, T):
        for t in range(1, T):
            data_cbf = sio.loadmat(prefix + f"cbf_{t}", matlab_compatible=True)["cbf_dat"]
            self.data_t[t] = (data_cbf['B'][0,0], data_cbf['deriv'][0,0], data_cbf['DtB'][0,0])
    
    def get_index(self, x):
        return np.array([
            np.argmin([np.abs(el - x[i]) for el in self.grid[i]]) for i in range(len(self.grid))
        ])

    def get_data_at_x(self, mat, x):
        x = x.reshape((x.shape[0],))
        x_new = [x[1], x[0], x[2]]
        return interpn(self.grid, mat, x_new, method='linear')[0]

    def B(self, x, t):
        return self.get_data_at_x(self.data_t[t][0], x)

    def dx(self, x, t):
        return np.array([[self.get_data_at_x(self.data_t[t][1][i,0], x) for i in range(self.data_t[t][1].shape[0])]]).T

    def dt(self, x, t):
        return self.get_data_at_x(self.data_t[t][2], x)

    def LguB(self, x,t):
         return self.dx(x, t).T @ self.g_u(x, t)

    def LgdB(self, x,t):
        return self.dx(x, t).T @ self.g_d(x, t)

    def LfB(self, x,t):
        return self.dx(x, t).T @ self.f(x, t)

class CBE():
    def __init__(self, cbf_u, cbf_d):
        self.cbf_u = cbf_u
        self.cbf_d = cbf_d
        self.grid = cbf_u.grid
        self.eq = None

    def eq_map(self, b_d, x, t):
        B_u = self.cbf_u.B(x, t)
        B_d = self.cbf_d.B(x, t)
        LfB_u = self.cbf_u.LfB(x, t)
        LfB_d = self.cbf_d.LfB(x, t)
        LguB_u = self.cbf_u.LguB(x, t)
        LgdB_u = self.cbf_u.LgdB(x, t)
        LguB_d = self.cbf_d.LguB(x, t)
        LgdB_d = self.cbf_d.LgdB(x, t)
        gamma_u = self.cbf_u.rate
        gamma_d = self.cbf_d.rate

        # Calculate sub-optimization value in constraint of main optimization
        d = cvx.Variable(LgdB_d.T.shape)
        sub_obj = cvx.Minimize(LfB_u + LgdB_d@d)
        sub_constraints = [LgdB_d@d >= b_d]
        if self.cbf_d.constraints is not None:
            sub_constraints.extend(self.cbf_d.constraints(d,x,t))
        sub_prob = cvx.Problem(sub_obj, sub_constraints)
        sub_val = sub_prob.solve(solver=cvx.GUROBI)

        if sub_val is None:
            return np.inf

        # Calculate main optimization value
        u = cvx.Variable(LguB_d.T.shape)
        obj = cvx.Minimize(LfB_d + LguB_d@u - gamma_d*B_d)
        constraints = [LguB_u@u >= gamma_u*B_u - sub_val]
        if self.cbf_u.constraints is not None:
            constraints.extend(self.cbf_u.constraints(u,x,t))
        prob = cvx.Problem(obj, constraints)
        val = prob.solve(solver=cvx.GUROBI)

        return -val

    def b_u_map(self, b_d, x, t):
        B_u = self.cbf_u.B(x, t)
        B_d = self.cbf_d.B(x, t)
        LfB_u = self.cbf_u.LfB(x, t)
        LfB_d = self.cbf_d.LfB(x, t)
        LguB_u = self.cbf_u.LguB(x, t)
        LgdB_u = self.cbf_u.LgdB(x, t)
        LguB_d = self.cbf_d.LguB(x, t)
        LgdB_d = self.cbf_d.LgdB(x, t)
        gamma_u = self.cbf_u.rate
        gamma_d = self.cbf_d.rate

        # Calculate main optimization value
        d = cvx.Variable(LguB_d.T.shape)
        obj = cvx.Minimize(LfB_u + LgdB_u@d - gamma_u*B_u)
        constraints = [LgdB_d@d >= b_d]
        if self.cbf_u.constraints is not None:
            constraints.extend(self.cbf_u.constraints(d,x,t))
        prob = cvx.Problem(obj, constraints)
        val = prob.solve(solver=cvx.GUROBI)

        return -val

    # Compute fixed point of eq_map
    def compute_b_fp(self, x, t, b_d0=None, method="grid"):
        if method == "grid":
            b_ds = np.arange(-10,10,0.01)
            dev_best = (self.eq_map(b_ds[0], x, t) - b_ds[0])**2
            b_d_best = b_ds[0]
            for b_d in b_ds:
                if b_d_best is None:
                    b_d_best = b_d
                    dev_best = (self.eq_map(b_d, x, t) - b_d)**2
                val = self.eq_map(b_d, x, t)
                dev = (val - b_d)**2
                if dev < 1e-5:
                    return b_d, dev
                if dev < dev_best:
                    dev_best = dev
                    b_d_best = b_d
            return b_d_best, dev_best
            # devs = np.array([(b-self.eq_map(b, x, t))**2 for b in b_ds])
            # return b_ds[np.argmin(devs)], np.min(devs)
        else:
            if b_d0 is None:
                b_d0 = 0
            return opt.fsolve(lambda b_d: self.eq_map(b_d, x, t), b_d0)

    def compute_b_fp_grid(self):
        b_fps = {}
        b_fps_tol = {}
        num = 0
        key_num = len(list(self.cbf_u.data_t.keys()))
        for t in self.cbf_u.data_t.keys():
            b_fp = np.zeros((self.grid[0].shape[0], self.grid[1].shape[0], self.grid[2].shape[0]))
            b_fp_tol = np.zeros((self.grid[0].shape[0], self.grid[1].shape[0], self.grid[2].shape[0]))
           
            for i in range(self.grid[0].shape[0]):
                for j in range(self.grid[1].shape[0]):
                    for k in range(self.grid[2].shape[0]):
                        num+=1
                        if num%100 == 0:
                            print(str(100*num/(key_num*self.grid[0].shape[0]*self.grid[1].shape[0]*self.grid[2].shape[0])) + "%", end=" ")
                        x = np.array([[self.grid[0][i], self.grid[1][j], self.grid[2][k]]]).T
                        b_fp[i,j,k], b_fp_tol[i,j,k] = self.compute_b_fp(x, t)
            b_fps[t] = b_fp
            b_fps_tol[t] = b_fp_tol
        self.b_fps = b_fps
        self.b_fps_tol = b_fps_tol

    def compute_eq(self):
        # Compute b_ds
        self.compute_b_fp_grid()

        # Compute b_us
        b_us = {}
        for t in self.cbf_u.data_t.keys():
            b_us = np.zeros((self.grid[0].shape[0], self.grid[1].shape[0], self.grid[2].shape[0]))
            for i in range(self.grid[0].shape[0]):
                for j in range(self.grid[1].shape[0]):
                    for k in range(self.grid[2].shape[0]):
                        b_d = self.b_fps[t][i,j,k]
                        x = np.array([[self.grid[0][i], self.grid[1][j], self.grid[2][k]]]).T
                        b_us[t][i,j,k] = self.b_u_map(self.b_fps[t], x, t)
        self.b_us = b_us

    def get_constraints(self, x, t):
        ind = self.get_index(x)
        b_d = self.b_fps[t][ind[0], ind[1], ind[2]]
        b_u = self.b_us[t][ind[0], ind[1], ind[2]]
        A_d = self.cbf_d.LgdB(x, t)
        A_u = self.cbf_u.LguB(x, t)
        return A_u, b_u, A_d, b_d

    def save(self, filename):
        data = {
            "grid": self.grid,
            "b_fps": self.b_fps,
            "b_fps_tol": self.b_fps_tol,
            "b_us": self.b_us,
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.grid = data["grid"]
        self.b_fps = data["b_fps"]
        self.b_fps_tol = data["b_fps_tol"]
        self.b_us = data["b_us"]
        self.cbe = data["cbe"]
    
    def get_index(self, x):
        return np.array([
            np.argmin([np.abs(el - x[i]) for el in self.grid[i]]) for i in range(len(self.grid))
        ])

    def get_fp(self, x, t):
        return self.b_fps[t][self.get_index(x)]

    def get_cbf_constraints(self, x, t):
        # Compute CBF constraints for worst case input. Want to compare size of control sets
        return None

    def plot_tol(self, t):
        for theta in range(self.b_fps_tol[t].shape[2]):
            plt.figure()
            plt.imshow(self.b_fps_tol[t][:,:,theta])
            plt.colorbar()
            plt.show()

def compute_volume(A, b):
    # Compute volume of polytope Ax <= b
    return None

    
        
    

