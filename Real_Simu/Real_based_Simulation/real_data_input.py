#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('../')
from main import *

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import multiprocessing as mp
import datetime
import pickle

np.random.seed(100)


def get_data_input(simulated_data):
    data_input = []
    class_list= simulated_data['class'].unique()
    for class_index in class_list:
        temp_df = simulated_data[simulated_data['class']==class_index]
        data_input.append(get_vector_of_class(temp_df))
    return data_input

def get_vector_of_class(temp_sub):
    temp_sub.sort_values(by='time_flag')
    list_of_i_t = []
    for index,row in temp_sub.iterrows():
        s_i_t = np.array([row['demand_cnt'],row['supply_cnt'],row['s_d_match_l1']])
        a_i_t = row['action']
        r_i_t = row['gmv']
        list_of_i_t.append([s_i_t,a_i_t,r_i_t])
    return list_of_i_t

def t_func(t):
    # the time state variable is an indicator for rush hours
    if(t%48 >= 14 and t%48 <= 44):
        return 1
    return 0
