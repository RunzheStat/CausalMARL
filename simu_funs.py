from utils import *
from weight import *
import main 
from simu_funs import *
import simu_DGP
reload(main)
reload(simu_DGP)


# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)
# tf.keras.backend.set_floatx('float64')

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


##########################################################################################################################################################


""" Generate a pattern and four target policy (also one no treatment policy) and run the simu_once for OPE_rep_times times.
"""
def simu(pattern_seed = 1,  l = 5, T = 14 * 24, thre_range = [9, 10, 11, 12, 13], u_D = None, # Setting - general
         OPE_rep_times = 20, n_cores = n_cores, inner_parallel = False, full_parallel = False,  # Parallel
         t_func = None, # DGP / target
         dim_S_plus_Ts = 3 + 3, epsilon = 1e-6,  # fixed
         sd_D = 3, sd_R = 0, sd_O = 1, sd_u_O = .3, # noises
         w_A = 1, w_O = .05, # Setting - spatial
         penalty = [.01, .01], penalty_NMF = None, CV_QV = False, # QV parameters
         n_layer = 2, w_hidden = 30, # NN structure
         Learning_rate = 1e-4, batch_size = 32, max_iteration = 1001, # NN training
         with_MF = False, with_NO_MARL = True, with_IS = True, 
        ): 
    """ generate the order pattern (used by all rep, shared by b and pi)
    """
    
    ####################################################################################################################################################################################
    npseed(pattern_seed)
    u_O = rnorm(100, sd_u_O, l**2) 
    
    print("max(u_O) = ", np.round(max(u_O), 1), "mean(u_O) = ", np.round(np.mean(u_O), 1))
        
    # generate the corresponding target plicy
    target_policys = []
    n_tp = len(thre_range)
    for i in range(n_tp):
        O_thre = thre_range[i]
        print("O_threshold = " + str(thre_range[i]))
        if i == 0: 
            target_policy = simu_DGP.simu_target_policy_pattern(l = l, u_O = u_O, threshold =  O_thre, print_flag = "all") # "all"
        else:
            target_policy = simu_DGP.simu_target_policy_pattern(l = l, u_O = u_O, threshold =  O_thre, print_flag = "None") # "policy_only"
        target_policys.append(target_policy)
        
    # generate the adj for the grid
    neigh = adj2neigh(getAdjGrid(l))
    
    ####################################################################################################################################################################################
    """ parallel
    """
    if full_parallel:
        # []_tp_1, ..., []_tp_n
        def once(seed):
            return simu_once(seed = seed % OPE_rep_times, l = l, T = T, t_func = t_func,  u_D = u_D,
                             u_O = u_O,  
                             target_policys = [target_policys[seed // OPE_rep_times]], w_A = w_A, w_O = w_O, dim_S_plus_Ts = dim_S_plus_Ts,  
                              penalty = penalty, penalty_NMF = penalty_NMF, 
                             n_layer = n_layer, sd_D = sd_D, sd_R = sd_R, sd_O = sd_O, 
                              w_hidden = w_hidden, Learning_rate = Learning_rate,  CV_QV = CV_QV, 
                              batch_size = batch_size, max_iteration = max_iteration, epsilon = epsilon, 
                             with_MF = with_MF, with_NO_MARL = with_NO_MARL, with_IS = with_IS, 
                               inner_parallel = inner_parallel, gpu_number = seed % 8)
        value_reps = parmap(once, range(OPE_rep_times * n_tp), n_cores)
        value_targets = []
        for i in range(len(target_policys)):
            value_targets.append([value[0] for value in value_reps[(i * OPE_rep_times):((i + 1) * OPE_rep_times)]])
        
    else:
        def once(seed):
            return simu_once(seed = seed, l = l, T = T, t_func = t_func,  u_D = u_D,
                             u_O = u_O,  
                             target_policys = target_policys, w_A = w_A, w_O = w_O, dim_S_plus_Ts = dim_S_plus_Ts,  
                              penalty = penalty, penalty_NMF = penalty_NMF, 
                             n_layer = n_layer, sd_D = sd_D, sd_R = sd_R, sd_O = sd_O, 
                              w_hidden = w_hidden, Learning_rate = Learning_rate,  CV_QV = CV_QV, 
                              batch_size = batch_size, max_iteration = max_iteration, epsilon = epsilon, 
                             with_MF = with_MF, with_NO_MARL = with_NO_MARL, with_IS = with_IS, 
                               inner_parallel = inner_parallel, gpu_number = seed % 8)
#         if not inner_parallel:
#             value_reps = parmap(once, range(OPE_rep_times), n_cores)
#         else:
        value_reps = []
        for seed in tqdm(range(OPE_rep_times)):
            print(seed)
            r = once(seed)
            value_reps.append(r)
        #value_reps = rep_seeds(once, OPE_rep_times)

        value_targets = []
        for i in range(len(target_policys)):
            value_targets.append([r[i] for r in value_reps])    
    ####################################################################################################################################################################################
    """ MC-based average reward following the target
    value_targets: len-n_target of len-100 of len-n_est results
    """
    good_setting_flag = 0
    print(Dash)
    
    def sd_mse_values(Vs, V_true):
        """ calculate the std of "the mse estimates"
        """
        n = len(Vs)
        temp = mean(Vs**4) - (mean(Vs**2))**2 + 4 * V_true ** 2 * var(Vs) - 4 * V_true * (mean(Vs**3) - mean(Vs**2) * mean(Vs))
        return sqrt(temp) / np.sqrt(n)
    
    def sd_mse_values_methods(V_OPE, V_MC):
        sd_mse = []
        for i in range(V_OPE.shape[1]):
            sd_mse.append(sd_mse_values(V_OPE[:, i], V_MC))
        return sd_mse

    Values_outputs_targets = []
    for i in range(len(target_policys)):
        V_MC, std_V_MC = MC_Value(l = l, T = T, t_func = t_func,  u_D = u_D, 
                                  u_O = u_O, sd_D = sd_D, sd_R = sd_R, sd_O = sd_O, w_A = w_A, w_O = w_O, 
                                  target_policy = target_policys[i], reps = 100, 
                                  inner_parallel = inner_parallel)
        V_OPE = value_targets[i]
        
        if i == 0:
            V_behav = np.mean(V_OPE, 0)[-1]
            print("Value of Behaviour policy:" + str(round(V_behav, 3)))
            
        print("O_threshold = " + str(thre_range[i]))
        print("MC for this TARGET:" + str([V_MC, std_V_MC]))
        
        """ Value
        """
        bias = np.round(np.nanmean(V_OPE, 0) - V_MC, 2) # nan
        std = np.round(np.nanstd(V_OPE, 0), 2)
        MSE = np.round(bias**2 + std**2, 2)
        RMSE = np.round(np.sqrt(bias**2 + std**2), 2)
        mse_rel = np.round(MSE - MSE[0], 2)
        sd_MSE = np.round(sd_mse_values_methods(arr(V_OPE), V_MC), 2)
        
        bias = list(bias); std = list(std); MSE = list(MSE); mse_rel = list(mse_rel)
        res = "   [DR/QV/IS]; [DR_NO_MARL, DR_NO_MF, V_behav]" + "\n" + \
        "bias:" + str([bias[:3]]) + str([bias[3:6]]) + "\n" + \
        "std:" + str([std[:3]]) + str([std[3:6]]) + "\n" + \
        "sd_MSE:" + str([sd_MSE[:3]]) + str([sd_MSE[3:6]])
        
        print(res)
        print("MSE:" + str([MSE[:3]]) + str([MSE[3:6]])) 
        print("RMSE(-DR):" + str([mse_rel[:3]]) + str([mse_rel[3:6]])) 

        Values_output = [arr(bias), arr(std), arr(MSE), arr(RMSE), sd_MSE, V_OPE, V_MC] # V_OPE = len-100 of len-n_est results
        Values_outputs_targets.append(Values_output)
                
    return Values_outputs_targets # a list (len-N_target) of list of [bias, std, MSE] (each is a vector). 


   ####################################################################################################################################################################################
def get_simu_results(value_targets, l, T, t_func, u_D
                    , u_O, sd_D, sd_R, sd_O, w_A, w_O
                    , inner_parallel):
    """ MC-based average reward following the target
    value_targets: len-n_target of len-100 of len-n_est results
    """
#     good_setting_flag = 0
#     print(Dash)
    
    def sd_mse_values(Vs, V_true):
        """ calculate the std of "the mse estimates"
        """
        n = len(Vs)
        temp = mean(Vs**4) - (mean(Vs**2))**2 + 4 * V_true ** 2 * var(Vs) - 4 * V_true * (mean(Vs**3) - mean(Vs**2) * mean(Vs))
        return sqrt(temp) / np.sqrt(n)
    
    def sd_mse_values_methods(V_OPE, V_MC):
        sd_mse = []
        for i in range(V_OPE.shape[1]):
            sd_mse.append(sd_mse_values(V_OPE[:, i], V_MC))
        return sd_mse

    Values_outputs_targets = []
    for i in range(len(value_targets)):
        V_MC, std_V_MC = MC_Value(l = l, T = T, t_func = t_func,  u_D = u_D, 
                                  u_O = u_O, sd_D = sd_D, sd_R = sd_R, sd_O = sd_O, w_A = w_A, w_O = w_O, 
                                  target_policy = target_policys[i], reps = 100, 
                                  inner_parallel = inner_parallel)
        V_OPE = value_targets[i]
        
        if i == 0:
            V_behav = np.mean(V_OPE, 0)[-1]
            print("Value of Behaviour policy:" + str(round(V_behav, 3)))
            
        print("O_threshold = " + str(thre_range[i]))
        print("MC for this TARGET:" + str([V_MC, std_V_MC]))
        
        """ Value
        """
        bias = np.round(np.nanmean(V_OPE, 0) - V_MC, 2) # nan
        std = np.round(np.nanstd(V_OPE, 0), 2)
        MSE = np.round(bias**2 + std**2, 2)
        RMSE = np.round(np.sqrt(bias**2 + std**2), 2)
        mse_rel = np.round(MSE - MSE[0], 2)
        sd_MSE = np.round(sd_mse_values_methods(arr(V_OPE), V_MC), 2)
        
        bias = list(bias); std = list(std); MSE = list(MSE); mse_rel = list(mse_rel)
        res = "   [DR/QV/IS]; [DR_NO_MARL, DR_NO_MF, V_behav]" + "\n" + \
        "bias:" + str([bias[:3]]) + str([bias[3:6]]) + "\n" + \
        "std:" + str([std[:3]]) + str([std[3:6]]) + "\n" + \
        "sd_MSE:" + str([sd_MSE[:3]]) + str([sd_MSE[3:6]])
        
        print(res)
        print("MSE:" + str([MSE[:3]]) + str([MSE[3:6]])) 
        print("RMSE(-DR):" + str([mse_rel[:3]]) + str([mse_rel[3:6]])) 

        Values_output = [arr(bias), arr(std), arr(MSE), arr(RMSE), sd_MSE, V_OPE, V_MC] # V_OPE = len-100 of len-n_est results
        Values_outputs_targets.append(Values_output)
                
    return Values_outputs_targets # a list (len-N_target) of list of [bias, std, MSE] (each is a vector). 


##############################################################################################################################################################################################################################################################################
def simu_once(seed = 1, l = 3, T = 14 * 24, t_func = None, u_D = None, 
              target_policys = None, 
              sd_D = 3, sd_R = 0, sd_O = 1, 
              CV_QV = False, 
              u_O = None, 
              dim_S_plus_Ts = 3 + 3,  w_A = 1, w_O = .05, 
              penalty = [1, 1], penalty_NMF = None, 
              n_layer = 2, 
              w_hidden = 10, Learning_rate = 1e-4,  
              batch_size = 64, max_iteration = 1001, epsilon = 1e-3, 
              with_MF = False, with_NO_MARL = True, with_IS  = True, 
               inner_parallel = False, gpu_number = 0, return_datail_only = False): 
    """
    Output: a len-n-target of len-est results
    """
    npseed(seed)
    print("seed = {}".format(seed))
    N = l ** 2
    def behav(s, a):
        return 0.5
    behav = list(itertools.repeat(behav, N))
    
    def Ts(S_neigh):
        return np.mean(S_neigh, 0)
    def Ta(A_neigh):
        return Ta_disc(np.mean(A_neigh, 0))
    ##############################################################################################################################################
    ### observed data following behav
    data, adj_mat, details = simu_DGP.DG_once(seed = seed, l = l, T = T, u_D = u_D, 
                                     u_O = u_O, 
                                     t_func = t_func,  
                                     sd_D = sd_D, sd_R = sd_R, sd_O = sd_O, 
                                     w_A = w_A, w_O = w_O)
    if return_datail_only:
        reg_data = main.V_DR(data = data, tp = target_policys[0], bp = behav, 
                                adj_mat = adj_mat, dim_S_plus_Ts = dim_S_plus_Ts, 
                             t_func = t_func, 
                             Ts = Ts, Ta = Ta, penalty = penalty, penalty_NMF = penalty_NMF, 
                                n_layer = n_layer, CV_QV = CV_QV, 
                            w_hidden = w_hidden, lr = Learning_rate,  
                            batch_size = batch_size, max_iteration = max_iteration, epsilon = epsilon, 
                                with_MF = with_MF, with_NO_MARL = with_NO_MARL, with_IS = with_IS, 
                             inner_parallel = inner_parallel, gpu_number = gpu_number, return_datail_only = return_datail_only)
        return data, adj_mat, details, reg_data
    ##############################################################################################################################################
    # OPE
    a = now()
    value_targets = []
    count = 0
    n_target = len(target_policys)
    for target_policy in target_policys:
        print("seed = {}, idx_target_policy = {} \n".format(seed, count))
        value_estimators = main.V_DR(data = data, tp = target_policy, bp = behav, 
                                adj_mat = adj_mat, dim_S_plus_Ts = dim_S_plus_Ts, 
                             t_func = t_func, 
                             Ts = Ts, Ta = Ta, penalty = penalty, penalty_NMF = penalty_NMF, 
                                n_layer = n_layer, CV_QV = CV_QV, 
                            w_hidden = w_hidden, lr = Learning_rate,  
                            batch_size = batch_size, max_iteration = max_iteration, epsilon = epsilon, 
                                with_MF = with_MF, with_NO_MARL = with_NO_MARL, with_IS = with_IS, 
                             inner_parallel = inner_parallel, gpu_number = gpu_number)
        count += 1
        value_targets.append(value_estimators)
        if seed == 0:
            print("target", count, "in", n_target, "DONE!")
    if inner_parallel and (seed + 1) % 5 == 0 :
        print("Rep", seed + 1, "DONE") 
    return value_targets

##########################################################################################################################################################
def MC_Value(l, T, t_func, target_policy, u_O = None, u_D = None, 
             sd_D =  3, sd_R = 0, sd_O = 1, 
             reps = 100, inner_parallel = False, w_A = 1, w_O = .01):
    """ Apply MC to get the target value
    """
    def oneTraj(seed):
        # Off-policy evaluation with target_policy
        data, adj_mat, details = simu_DGP.DG_once(seed = seed, l = l, T = T, t_func = t_func, u_D = u_D, 
                                          u_O = u_O, sd_O = sd_O, 
                                         TARGET = True, target_policy = target_policy, 
                                         sd_D = sd_D, sd_R = sd_R, w_A = w_A, w_O = w_O)
        R = details[2]
        V = np.mean(R)
        return V
    if inner_parallel:
        Vs = parmap(oneTraj, range(reps))
    else:
        Vs = rep_seeds(oneTraj, reps)
    # std among reps is small
    res = np.round([np.mean(Vs), np.std(Vs)], 3)  
    return res
