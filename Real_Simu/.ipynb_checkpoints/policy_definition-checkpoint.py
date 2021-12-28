#!/usr/bin/env python
# coding: utf-8

from real_data_input import *
np.random.seed(100)

def behav(s, a):
    return 0.5
bp = list(itertools.repeat(behav, 8))

fixed_policy_1 = list(itertools.repeat(1, 8))
pi_1 = []
for reward in fixed_policy_1:
    def pi_i(s = None, a = 0, random_choose = False, reward = reward):
        if random_choose:
            return reward
        else:
            return int(a == reward)
    pi_1.append(pi_i)
    
fixed_policy_0 = list(itertools.repeat(0, 8))
pi_0 = []
for reward in fixed_policy_0:
    def pi_i(s = None, a = 0, random_choose = False, reward = reward):
        if random_choose:
            return reward
        else:
            return int(a == reward)
    pi_0.append(pi_i)

# target policy
adaptive_action_list_t = []
adaptive_tp_dictionary_t = []
for region_index in range(8):
    region_action=[]
    for time_index in range(48):
        if(time_index%48 >= 14 and time_index%48 <= 44):
            adaptive_action_list_t.append(1)
            region_action.append(1)
        else:
            adaptive_action_list_t.append(0)
            region_action.append(0)
    adaptive_tp_dictionary_t.append(region_action)

pi_adaptive_t = []
for i in range(8):
    def tp_i(s, a = 0, random_choose = False, fixed_policy_i = adaptive_tp_dictionary_t[i]):
        t = int(s[3])
        if random_choose:
            return int(fixed_policy_i[t])
        else:
            return int(a == fixed_policy_i[t])
    pi_adaptive_t.append(tp_i)

    
adaptive_action_list_ts = []
adaptive_tp_dictionary_ts = []
for region_index in range(8):
    region_action=[]
    for time_index in range(48):
        if(time_index%48 >= 14 and time_index%48 <= 44 and region_index in [4,6,7]):
            adaptive_action_list_ts.append(1)
            region_action.append(1)
        else:
            adaptive_action_list_ts.append(0)
            region_action.append(0)
    adaptive_tp_dictionary_ts.append(region_action)

pi_adaptive_ts = []
for i in range(8):
    def tp_i(s, a = 0, random_choose = False, fixed_policy_i = adaptive_tp_dictionary_ts[i]):
        t = int(s[3])
        if random_choose:
            return int(fixed_policy_i[t])
        else:
            return int(a == fixed_policy_i[t])
    pi_adaptive_ts.append(tp_i)
    

adaptive_action_list_s = []
adaptive_tp_dictionary_s = []
for region_index in range(8):
    region_action=[]
    for time_index in range(48):
        if(region_index in [4,6,7]):
            adaptive_action_list_s.append(1)
            region_action.append(1)
        else:
            adaptive_action_list_s.append(0)
            region_action.append(0)
    adaptive_tp_dictionary_s.append(region_action)

pi_adaptive_s = []
for i in range(8):
    def tp_i(s, a = 0, random_choose = False, fixed_policy_i = adaptive_tp_dictionary_s[i]):
        t = int(s[3])
        if random_choose:
            return int(fixed_policy_i[t])
        else:
            return int(a == fixed_policy_i[t])
    pi_adaptive_s.append(tp_i)
