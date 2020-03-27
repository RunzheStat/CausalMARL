#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#############################################################################
#############################################################################
#%% packages
# last project
from _uti_basic import *
from scipy.linalg import sqrtm
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.neighbors import KernelDensity

from itertools import combinations 

import operator
import time
now = time.time
from sklearn.model_selection import KFold

from numpy import squeeze
from statsmodels.stats import proportion as prop
import os, itertools
from scipy.spatial.distance import pdist as pdist

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

from numpy.random import lognormal as rlogN
#############################################################################
from sklearn.metrics.pairwise import rbf_kernel as GRBF
from numpy.linalg import solve
#############################################################################
#############################################################################

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
        if ta <= 2/8:
            return 0
        if ta <= 5/8:
            return 1
        else:
            return 2

#############################################################################
def getAdjGrid(l):
    """
    simple: only 4 neigh
    
    """
    simple = True
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


    
