#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#############################################################################
# Packages
import scipy as sp
import sklearn as sk
from importlib import reload
from tqdm import tqdm
# import ray
# Random
from numpy.random import seed as npseed
from numpy import absolute as np_abs
from numpy.random import normal as rnorm
from numpy.random import uniform as runi
from numpy.random import binomial as rbin
from numpy.random import poisson as rpoisson
from numpy.random import shuffle,randn, permutation # randn(d1,d2) is d1*d2 i.i.d N(0,1)
from numpy.random import lognormal as rlogN
from numpy import squeeze
from numpy.linalg import solve

# Numpy
import numpy as np
from numpy import mean, var, std, median
from numpy import array as arr
from numpy import sqrt, cos, sin, exp, dot, diag, ones, identity, quantile, zeros, roll, multiply, stack, concatenate
from numpy import concatenate as v_add
from numpy.linalg import norm, inv
from numpy import apply_along_axis as apply

from zipfile import ZipFile

def unzip(path, zip_type = "tar_gz"):
    if zip_type == "tar_gz":
        import tarfile
        tar = tarfile.open(path, "r:gz")
        tar.extractall()
        tar.close()
    elif zip_type == "zip":        
        with ZipFile(path, 'r') as zipObj:
           # Extract all the contents of zip file in current directory
           zipObj.extractall()


import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from scipy.spatial.distance import pdist as pdist
from scipy.stats import binom


from sklearn.model_selection import KFold
from sklearn import preprocessing as pre
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from sklearn.metrics.pairwise import rbf_kernel as GRBF


np.set_printoptions(precision = 4)
#############################################################################
import time
now = time.time


#%% packages
import os, itertools
from itertools import combinations 

import time
now = time.time

#############################################################################
dash = "--------------------------------------"
DASH = "\n" + "--------------------------------------" + "\n"
Dash = "\n" + dash
dasH = dash + "\n"
#############################################################################
#%% utility funs
from multiprocessing import Pool
import multiprocessing
n_cores = multiprocessing.cpu_count()

def mute():
    sys.stdout = open(os.devnull, 'w')    

def rep_seeds(fun,rep_times):
    """
    non-parallel-version of pool.map
    """
    return list(map(fun, range(rep_times)))

def rep_seeds_print(fun, rep_times, init_seed = 0):
    r = []
    start = now()
    for seed in range(rep_times):
        r.append(fun(seed + init_seed))
        if seed % 25 == 0:
            print(round((seed+1)/rep_times*100,2),"% DONE", round((now() - start)/60,2), "mins" )
    return r
def listinlist2list(theList):
    return [item for sublist in theList for item in sublist]

def is_disc(v, n):
    return len(set(v)) <= n

def R2(y_true, y_pred):
    unexplained = np.mean((y_true - y_pred)**2) 
    true_var = np.mean((y_true - np.mean(y_true))**2)
#     print("true_var:", true_var, "unexplained:", unexplained )
    return 1 - unexplained / true_var
        
def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))
        
def parmap(f, X, nprocs = multiprocessing.cpu_count()):#-2
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]

def round_list(thelist,dec):
    """
    extend np.round to list
    """
    return [round(a,dec) for a in thelist]

def print_time_cost(seed,total_rep,time):
    print(round((seed+1/total_rep)*100,3),"% DONE, takes", round((time)/60,3)," mins \n")

simple = False

##########################################################################################################################################################

def adj2neigh(adj_mat):
    neigh = {}
    N = adj_mat.shape[0]
    for i in range(N):
        temp = []
        for j in range(N):
            if j != i and adj_mat[i][j] == 1:
                temp.append(j)
        neigh[i] = temp
    return neigh


def Ta_disc(ta):
    # regardless of n_neigh
    if simple:
        if ta <= 1/4:
            return 0
        if ta == 2/4:
            return 1
        else:
            return 2
#         return ta
    else:
        if ta <= 2/8:
            return 0
        if ta <= 5/8:
            return 1
        else:
            return 2

def den_b_disc(Ta, N_neigh): 
    # N_neigh is not a constant
    den = 0
    for i in range(N_neigh + 1):
        if Ta_disc(i / N_neigh) == Ta:
            den += binom.pmf(i, N_neigh, 0.5)
    return den

##########################################################################################################################################################
def getAdjGrid(l):
    """
    simple: only 4 neigh
    
    """
    N = l ** 2
    adj_mat = zeros((N, N))
    for i in range(N):
        row = i // l
        col = i % l
        adj_mat[i][i] = 1
        if row != 0:
            adj_mat[i][i - l] = 1
            if not simple:
                if col != 0:
                    adj_mat[i][i - l - 1] = 1
                if col != l - 1:
                    adj_mat[i][i - l + 1] = 1
        if row != l - 1:
            adj_mat[i][i + l] = 1
            if not simple:
                if col != 0:
                    adj_mat[i][i + l - 1] = 1
                if col != l - 1:
                    adj_mat[i][i + l + 1] = 1
        if col != 0:
            adj_mat[i][i - 1] = 1
        if col != l - 1:
            adj_mat[i][i + 1] = 1
    return adj_mat


def t_func_peri(t):
    return t%48

#############################################################################
#############################################################################
import inspect
import functools

def autoargs(*include, **kwargs):
    def _autoargs(func):
        attrs, varargs, varkw, defaults = inspect.getargspec(func)

        def sieve(attr):
            if kwargs and attr in kwargs['exclude']:
                return False
            if not include or attr in include:
                return True
            else:
                return False

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # handle default values
            if defaults:
                for attr, val in zip(reversed(attrs), reversed(defaults)):
                    if sieve(attr):
                        setattr(self, attr, val)
            # handle positional arguments
            positional_attrs = attrs[1:]
            for attr, val in zip(positional_attrs, args):
                if sieve(attr):
                    setattr(self, attr, val)
            # handle varargs
            if varargs:
                remaining_args = args[len(positional_attrs):]
                if sieve(varargs):
                    setattr(self, varargs, remaining_args)
            # handle varkw
            if kwargs:
                for attr, val in kwargs.items():
                    if sieve(attr):
                        setattr(self, attr, val)
            return func(self, *args, **kwargs)
        return wrapper
    return _autoargs
#############################################################################
#############################################################################
# pd.options.display.max_rows = 10
