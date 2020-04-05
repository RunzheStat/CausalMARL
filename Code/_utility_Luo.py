#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#############################################################################

#%% packages
from _uti_basic import *
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity

from itertools import combinations 

import operator
import time
now = time.time
from statsmodels.stats import proportion as prop
import os, itertools
from scipy.spatial.distance import pdist as pdist

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

from numpy.random import lognormal as rlogN

from sklearn.metrics.pairwise import rbf_kernel as GRBF
from scipy.stats import binom


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


def Ta_disc(ta, simple = False):
    # regardless of n_neigh
    if not simple: 
        if ta <= 2/8:
            return 0
        if ta <= 5/8:
            return 1
        else:
            return 2
    else: # may be do not need discrete
        if ta <= 1/4:
            return 0
        if ta == 2/4:
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
def getAdjGrid(l, simple = False):
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


#############################################################################
# Packages
import scipy as sp
import pandas as pd
import sklearn as sk

# Random
from random import seed as rseed
from numpy.random import seed as npseed
from numpy import absolute as np_abs
from numpy.random import normal as rnorm
from numpy.random import uniform as runi
from numpy.random import binomial as rbin
from numpy.random import poisson as rpoisson
from numpy.random import shuffle,randn, permutation # randn(d1,d2) is d1*d2 i.i.d N(0,1)

# Numpy
import numpy as np
from numpy import array as arr
from numpy import sqrt, cos, sin, exp, dot, diag, ones, identity, quantile, zeros, roll, multiply, stack, concatenate
from numpy import concatenate as v_add
from numpy.linalg import norm, inv
from numpy import apply_along_axis as apply
from sklearn import preprocessing as pre

from termcolor import colored, cprint
from matplotlib.pyplot import hist
import pickle
from sklearn.model_selection import GridSearchCV

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


from sklearn.model_selection import KFold
from numpy import squeeze
from numpy.linalg import solve

np.set_printoptions(precision = 4)
#############################################################################
import time
now = time.time
import smtplib, ssl

import datetime, pytz

def EST():
    return datetime.datetime.now().astimezone(pytz.timezone('US/Eastern')).strftime("%H:%M, %m/%d")

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

def send_email(message = None, email_address = "13300180059@fudan.edu.cn", title = "Your results are ready!",
              receiver_email = "Same"): # py.notify.me@gmail.com
    port = 465  # For SSL
    # Create a secure SSL context
    context = ssl.create_default_context()
    sender_email = email_address # "py.notify.me@gmail.com"
    if receiver_email == "Same":
        receiver_email = email_address
    email_content = message
    
    a = """

    """
    
    message = """\
    Subject: """ + title + a
    message += email_content
    
    with smtplib.SMTP_SSL("mail.fudan.edu.cn", port, context=context) as server: # "smtp.gmail.com"
        server.login(email_address,"w19950722")  #("py.notify.me@gmail.com", "w19950722")
        server.sendmail(sender_email, receiver_email, message)

#############################################################################
# https://pypi.org/project/termcolor/#description
def printR(theStr):
    print(colored(theStr, 'red'))
          
def printG(theStr):
    print(colored(theStr, 'green'))
          
def printB(theStr):
    print(colored(theStr, 'blue'))
#############################################################################
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
#############################################################################     

def round_list(thelist,dec):
    """
    extend np.round to list
    """
    return [round(a,dec) for a in thelist]

def print_time_cost(seed,total_rep,time):
    print(round((seed+1/total_rep)*100,3),"% DONE, takes", round((time)/60,3)," mins \n")
    
def is_disc(v, n):
    return len(set(v)) <= n

def R2(y_true, y_pred):
    unexplained = np.mean((y_true - y_pred)**2) 
    true_var = np.mean((y_true - np.mean(y_true))**2)
#     print("true_var:", true_var, "unexplained:", unexplained )
    return 1 - unexplained / true_var

def print_progress(i, N, freq = 100):
    if (i * freq // N == 0):
        print("#", end = "", flush = True)

        
def rangeofVec(v, precision = 2):
    # np.mean(), np.std()
    return [np.round(np.quantile(v, [0.01, 0.1, 0.5, 0.9, 0.99]), precision),  np.round(max(abs(v)), precision)]

def uv(v, precision = 3):
    return np.round([np.mean(v), np.std(v)], precision)