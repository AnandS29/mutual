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

# Dynamics
v_e, v_p = 5, 5
f = lambda x,t: np.array([[-v_e + v_p*np.cos(x[2,0]), v_p*np.sin(x[2,0]), 0]]).T
g_u = lambda x,t: np.array([[x[1,0], -x[0,0], -1]]).T
g_d = lambda x,t: np.array([[0, 0, 1]]).T

# Grid
grid = (np.linspace(-7,7,20),np.linspace(-7,7,20),np.linspace(-np.pi,np.pi,5)) # np.mgrid[-7:7:(14/19), -7:7:(14/19), -np.pi:np.pi:(2*np.pi/4)]

# Time
ts = np.arange(0, .9, 0.01)

# Make CBF
prefix = '/Users/anandsranjan/Desktop/hybrid-workspace/CBVF/cbf_clf_helper_with_tv/demos/dubins_car/cbf_data/'
T = len(ts)
constraints = lambda u,x,t: [u <= 3, u >= -3]
cbf_u = CBF(f, g_u, g_d, grid, constraints=constraints)
cbf_u.load_data(prefix, T)
cbf_d = CBF(f, g_d, g_u, grid, constraints=constraints)
cbf_d.load_data(prefix, T)

# Make CBE
cbe = CBE(cbf_u, cbf_d)

# Test eq_map
x = np.array([[0, 0, 0]]).T
t = 1
b_d = 0
cbe.eq_map(b_d, x, t)

# Test compute_b_fp
x = np.array([[0, 0, 0]]).T
t = 1
b_d0 = 0
cbe.compute_b_fp(x, t, b_d0)

cbe.eq_map(-5.630000000000093, x, t)

# Compute all eq
cbe.compute_eq()
cbe.save('cbe_data.pkl')