#!/usr/bin/env python
# coding: utf-8

# Get result dataframe from the result "result.pkl"

import pandas as pd
import numpy as np
import pickle


def get_df_result(result_list):
    result_df = pd.DataFrame()
    
    DR_V_ada_t=[x[0][0] for x in result_list]
    QV_V_ada_t=[x[0][1] for x in result_list]
    IS_V_ada_t=[x[0][2] for x in result_list]
    DR_V_NS_ada_t=[x[0][3] for x in result_list]
    DR_V_NMF_ada_t=[x[0][4] for x in result_list]
    V_behav_ada_t=[x[0][5] for x in result_list]
    
    DR_V_ada_ts=[x[1][0] for x in result_list]
    QV_V_ada_ts=[x[1][1] for x in result_list]
    IS_V_ada_ts=[x[1][2] for x in result_list]
    DR_V_NS_ada_ts=[x[1][3] for x in result_list]
    DR_V_NMF_ada_ts=[x[1][4] for x in result_list]
    V_behav_ada_ts=[x[1][5] for x in result_list]
    
    DR_V_ada_s=[x[2][0] for x in result_list]
    QV_V_ada_s=[x[2][1] for x in result_list]
    IS_V_ada_s=[x[2][2] for x in result_list]
    DR_V_NS_ada_s=[x[2][3] for x in result_list]
    DR_V_NMF_ada_s=[x[2][4] for x in result_list]
    V_behav_ada_s=[x[2][5] for x in result_list]
    
    ground_truth_ada_t=[x[3] for x in result_list]
    ground_truth_ada_ts=[x[4] for x in result_list]
    ground_truth_ada_s=[x[5] for x in result_list]
    
    result_df['DR_V_ada_t'] = DR_V_ada_t
    result_df['QV_V_ada_t'] = QV_V_ada_t
    result_df['IS_V_ada_t'] = IS_V_ada_t
    result_df['DR_V_NS_ada_t'] = DR_V_NS_ada_t
    result_df['DR_V_NMF_ada_t'] = DR_V_NMF_ada_t
    result_df['V_behav_ada_t'] = V_behav_ada_t
    
    result_df['DR_V_ada_ts'] = DR_V_ada_ts
    result_df['QV_V_ada_ts'] = QV_V_ada_ts
    result_df['IS_V_ada_ts'] = IS_V_ada_ts
    result_df['DR_V_NS_ada_ts'] = DR_V_NS_ada_ts
    result_df['DR_V_NMF_ada_ts'] = DR_V_NMF_ada_ts
    result_df['V_behav_ada_ts'] = V_behav_ada_ts
    
    result_df['DR_V_ada_s'] = DR_V_ada_s
    result_df['QV_V_ada_s'] = QV_V_ada_s
    result_df['IS_V_ada_s'] = IS_V_ada_s
    result_df['DR_V_NS_ada_s'] = DR_V_NS_ada_s
    result_df['DR_V_NMF_ada_s'] = DR_V_NMF_ada_s
    result_df['V_behav_ada_s'] = V_behav_ada_s
    
    result_df['ground_truth_ada_t'] = ground_truth_ada_t
    result_df['ground_truth_ada_ts'] = ground_truth_ada_ts
    result_df['ground_truth_ada_s'] = ground_truth_ada_s
    
    return result_df


methods = ['DR_V','QV_V','IS_V','DR_V_NS','DR_V_NMF','V_behav','ground_truth']

def get_mean_sq(temp_list,ground_truth):
    sum_x = 0
    N = len(temp_list)
    for x in temp_list:
        sum_x = sum_x + pow((x-ground_truth),2)
    return round(sum_x/N,5)

def get_statistic_result(temp_result_df,map_name):
    data_policy = map_name.split('_')[1]
    if('ada' in data_policy):
        data_policy = map_name.split('_')[1]+map_name.split('_')[2]
    index_name = map_name
    est_list = list(round(np.mean(temp_result_df)[methods],5))
    bias_list = list(round(np.mean(temp_result_df)[methods]-np.mean(temp_result_df)['ground_truth'],5))
    std_list = list(round(np.std(temp_result_df),5)[methods])
    msq_list = [get_mean_sq(list(temp_result_df[name]),np.mean(temp_result_df['ground_truth'])) for name in methods]
    data_array = np.array([est_list,bias_list,std_list,msq_list])
    test_df = pd.DataFrame(data = data_array, columns=methods,                           index = [[index_name,index_name,index_name,index_name],                                    [data_policy,data_policy,data_policy,data_policy],                                    ['value','bias','std','mse']])
    return test_df



names_ada_t = ['DR_V_ada_t','QV_V_ada_t','IS_V_ada_t','DR_V_NS_ada_t','DR_V_NMF_ada_t','V_behav_ada_t','ground_truth_ada_t']
names_ada_ts = ['DR_V_ada_ts','QV_V_ada_ts','IS_V_ada_ts','DR_V_NS_ada_ts','DR_V_NMF_ada_ts','V_behav_ada_ts','ground_truth_ada_ts']
names_ada_s = ['DR_V_ada_s','QV_V_ada_s','IS_V_ada_s','DR_V_NS_ada_s','DR_V_NMF_ada_s','V_behav_ada_s','ground_truth_ada_s']



file = open('result.pkl', 'rb')
result_list_3 = pickle.load(file)
file.close()
result_df = get_df_result(result_list_3)
result = result_df


data_map = {}

data_map['names_ada_s'] = result[names_ada_s]
data_map['names_ada_t'] = result[names_ada_t]
data_map['names_ada_ts'] = result[names_ada_ts]


all_result=[]
for key,result_df in data_map.items():
    result_df.columns = ['DR_V','QV_V','IS_V','DR_V_NS','DR_V_NMF','V_behav','ground_truth']
    temp_result = get_statistic_result(result_df,key)
    all_result.append(temp_result)
all_result_df = pd.concat(all_result)   

print(all_result_df[['DR_V','IS_V','DR_V_NS']])