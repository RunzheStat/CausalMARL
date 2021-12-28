import sys
sys.paths.append('../src/')

import pandas as pd
from real_data_input import *
from policy_definition import *
np.random.seed(100)

his_est = pd.read_csv('data/history_estimator_data.csv')
his_est = his_est[['class','time_flag','demand_cnt_his','supply_cnt_his']]

adj_matrix = np.array([[1., 0., 1., 0., 1., 0., 0., 1.],
       [0., 1., 0., 0., 0., 1., 1., 0.],
       [1., 0., 1., 1., 0., 0., 0., 1.],
       [0., 0., 1., 1., 0., 1., 0., 1.],
       [1., 0., 0., 0., 1., 0., 0., 1.],
       [0., 1., 0., 1., 0., 1., 1., 1.],
       [0., 1., 0., 0., 0., 1., 1., 1.],
       [1., 0., 1., 1., 1., 1., 1., 1.]])
class_num = 8
area_list = [434, 298, 442, 328, 804, 354, 455, 482]

rds = RealDataSimulation(class_num,adj_matrix,area_list)
rds.std_value_map_log = {'demand_cnt': 1.082645635915417,\
                         'supply_cnt': 1.180323154271419,\
                         'gmv': 1.835183691515375,\
                         's_d_match': 0.5506578267576276}
rds.std_value_map = {'demand_cnt': 26.735751851427977,
                     'supply_cnt': 26.039399744132286,
                     'gmv': 928.8104517320271,
                     's_d_match': 2.6695942370348043}
rds.beta_last = [12.3715, 32.9114, 4.5717, 39.6882, -8.0347, 16.4557, 2.2364, 12.799 , -4.474 ]
rds.his_estimator = his_est
pen_param = [[1e-4,1e-1], [1e-4,1e-1]]

def get_simulated_data(x):
    [day_num, policy, action_promote_ratio] = list(x)
    simulated_data = rds.get_simulated_data_with_gmv(day_num, policy, action_promote_ratio)
    return simulated_data

def get_V_DR_result(action_promote_ratio=0.05,week_num = 2):
    params = list(itertools.repeat([1, adaptive_action_list_t, action_promote_ratio], int(week_num*7)))+\
             list(itertools.repeat([1, adaptive_action_list_ts,action_promote_ratio], int(week_num*7)))+\
             list(itertools.repeat([1, adaptive_action_list_s, action_promote_ratio], int(week_num*7)))+\
             list(itertools.repeat([1, 'random',               action_promote_ratio], int(week_num*7)))
    
    pool = mp.Pool(mp.cpu_count()-3)
    data_list = pool.map(get_simulated_data, params)
    pool.close()

    ground_truth_adaptive_t_df  = pd.concat(data_list[:int(week_num*7)])
    ground_truth_adaptive_ts_df = pd.concat(data_list[int(week_num*7):int(2*week_num*7)])
    ground_truth_adaptive_s_df  = pd.concat(data_list[int(2*week_num*7):int(3*week_num*7)])
    
    ground_truth_adaptive_t  = np.mean(ground_truth_adaptive_t_df['gmv'])
    ground_truth_adaptive_ts = np.mean(ground_truth_adaptive_ts_df['gmv'])
    ground_truth_adaptive_s  = np.mean(ground_truth_adaptive_s_df['gmv'])
    
    simulated_behavior_data = pd.concat(data_list[int(3*week_num*7):int(4*week_num*7)])
    
    adj_matrix = rds.adj_matrix
    input_data_behavior = get_data_input(simulated_behavior_data)
    print('begin V_DR')
    
    result_of_pi_adaptive_t = V_DR(input_data_behavior,adj_matrix,pi_adaptive_t,bp,
                                   None,None,t_func=t_func,penalty=pen_param,CV_QV=True,\
                                   with_MF = False,with_NO_MARL=True,inner_parallel=True)
    print('done adaptive t---------------------------------------')
    result_of_pi_adaptive_ts = V_DR(input_data_behavior,adj_matrix,pi_adaptive_ts,bp,
                                    None,None,t_func=t_func,penalty=pen_param,CV_QV=True,\
                                    with_MF = False,with_NO_MARL=True,inner_parallel=True)
    print('done adaptive ts---------------------------------------')
    result_of_pi_adaptive_s = V_DR(input_data_behavior,adj_matrix,pi_adaptive_s,bp,
                                   None,None,t_func=t_func,penalty=pen_param,CV_QV=True,\
                                   with_MF = False,with_NO_MARL=True,inner_parallel=True)
    print('done adaptive s---------------------------------------')
    return [result_of_pi_adaptive_t,result_of_pi_adaptive_ts,result_of_pi_adaptive_s,\
            ground_truth_adaptive_t,ground_truth_adaptive_ts,ground_truth_adaptive_s]

result_list =[]
for seed in range(0,100):
    np.random.seed(seed)
    print(seed,' times')
    result = get_V_DR_result(action_promote_ratio=0.1, week_num=2)
    print(seed,' done ----------------------------')
    result_list.append(result)
    file = open('result/result.pkl', 'wb')
    pickle.dump(result_list, file)
    file.close()