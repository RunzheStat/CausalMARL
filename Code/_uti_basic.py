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

from sklearn.model_selection import GridSearchCV

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

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
        
def parmap(f, X, nprocs=multiprocessing.cpu_count()-2):
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

def print_progress(i, N):
    if (i * 100 // N == 0):
        print("#", end = "", flush = True)

        
def rangeofVec(v, precision = 2):
    # np.mean(), np.std()
    return [np.round(np.quantile(v, [0.01, 0.1, 0.5, 0.9, 0.99]), precision),  np.round(max(abs(v)), precision)]

def uv(v, precision = 3):
    return np.round([np.mean(v), np.std(v)], precision)