from _uti_basic import *
from _utility import *
from weight import *
from main import *
from simu_funs import *
from simu_DGP import *


##########################################################################################################################################################


""" Generate a pattern and four target policy (also one no treatment policy) and run the simu_once for OPE_rep_times times.
"""
def simu(pattern_seed = 1,  l = 5, T = 14 * 24, thre_range = [9, 10, 11, 12, 13], # Setting - general
         OPE_rep_times = 20, n_cores = n_cores, inner_parallel = False,  # Parallel
         time_dependent = False, # DGP / target
         dim_S_plus_Ts = 3 + 3, epsilon = 1e-6,  # fixed
         sd_D = 3, sd_R = 0, sd_O = 1, sd_u_O = .3, # noises
         w_A = 1, w_O = .05, # Setting - spatial
         penalty = [.01, .01], penalty_NMF = None, CV_QV = False, # QV parameters
         n_layer = 2, w_hidden = 30, # NN structure
         Learning_rate = 1e-4, batch_size = 32, max_iteration = 1001, # NN training
           # debug
         file = None, print_flag_target = True # echo
        ): 
    """ generate the order pattern (used by all rep, shared by b and pi)
    """

    npseed(pattern_seed)
    u_O = rlogN(2.4, sd_u_O, l**2) 
#     u_O = rlogN(5, sd_u_O, l**2)
    print("max(u_O) = ", max(u_O))
        
    # generate the corresponding target plicy
    target_policys = []
    for i in range(len(thre_range)):
        O_thre = thre_range[i]
        printG("O_threshold = " + str(thre_range[i]))
        if i == 0:#, n_r 
            target_policy = simu_target_policy_pattern(u_O = u_O, threshold =  O_thre, print_flag = "all")
        else:
            target_policy = simu_target_policy_pattern(u_O = u_O, threshold =  O_thre, print_flag = "policy_only")
        target_policys.append(target_policy)
        
    # generate the adj for the grid
    neigh = adj2neigh(getAdjGrid(l, simple = simple))
    
    """ parallel
    """
    def once(seed):
        return simu_once(seed = seed, l = l, T = T, time_dependent = time_dependent,  
                         u_O = u_O,  
                         target_policys = target_policys, w_A = w_A, w_O = w_O, dim_S_plus_Ts = dim_S_plus_Ts, n_cores = n_cores, 
                          penalty = penalty, penalty_NMF = penalty_NMF, 
                         n_layer = n_layer, sd_D = sd_D, sd_R = sd_R, sd_O = sd_O, 
                          w_hidden = w_hidden, Learning_rate = Learning_rate,  CV_QV = CV_QV, 
                          batch_size = batch_size, max_iteration = max_iteration, epsilon = epsilon, 
                           inner_parallel = inner_parallel)
    if not inner_parallel:
        value_reps = parmap(once, range(OPE_rep_times), n_cores)
    else:
        value_reps = rep_seeds(once, OPE_rep_times)
    
    value_targets = []
    for i in range(len(target_policys)):
        value_targets.append([r[i] for r in value_reps])
    
    """ MC-based average reward following the target
    """
    good_setting_flag = 0
    print(Dash)
    for i in range(len(target_policys)):
        printG("O_threshold = " + str(thre_range[i]))
        target_policy = target_policys[i]
        V_MC, std_V_MC = MC_Value(l = l, T = T, time_dependent = time_dependent,  
                                  u_O = u_O, sd_D = sd_D, sd_R = sd_R, sd_O = sd_O, w_A = w_A, w_O = w_O, 
                                  target_policy = target_policy, reps = 100, 
                                  inner_parallel = inner_parallel)
        V_OPE = value_targets[i]
        if i == 0:
            V_behav = np.mean(V_OPE, 0)[-1]
            printG("Value of Behaviour policy:" + str(round(V_behav, 3)))
        
        """ Value estimation
        """
        bias = np.round(np.abs(np.mean(V_OPE, 0) - V_MC), 2)
    #     bias = np.round(np.mean(V_OPE, 0) - V_MC, 2)
        std = np.round(np.std(V_OPE, 0), 2)
        mse = np.round(np.sqrt(bias**2 + std**2), 2)
        mse_rel = np.round(mse - mse[0], 2)
        bias = list(bias); std = list(std); mse = list(mse); mse_rel = list(mse_rel) 
        res = "   [DR/QV/IS]; [DR/QV/IS]_NO_MARL; [DR/QV/IS]_NO_MF; [DR2, V_behav]" + "\n" + "bias:" + str([bias[:3]]) + str([bias[3:6]]) + str([bias[6:9]]) + str([bias[9:]]) + "\n" + "std:" + str([std[:3]]) + str([std[3:6]]) + str([std[6:9]]) + str([std[9:]]) + "\n" + \
                "MSE:" + str([mse[:3]]) + str([mse[3:6]]) + str([mse[6:9]]) + str([mse[9:]])
        print(res)
        printB("MSE(-DR):" + str([mse_rel[:3]]) + str([mse_rel[3:6]]) + str([mse_rel[6:9]]) + str([mse_rel[9:]]))# + "\n"
        if file is not None:        
            print(res + "\n" + "MSE(-DR):" + str([mse_rel[:3]]) + str([mse_rel[3:6]]) + str([mse_rel[6:9]]) + str([mse_rel[9:]]) + "\n", file = file)

        if mse_rel[0] <= np.min(mse_rel[2:4]):
            cprint("***** BETTER THAN [QV, IS, DR_NO_MARL] *****", 'white', 'on_red')
            print("*****BETTER THAN [QV, IS, DR_NO_MARL]*****", file = file)
            good_setting_flag += 1
        elif mse_rel[0] <= np.min(mse_rel[3:6]):  
            cprint("better than DR_NO_MARL", "yellow")
            print("better than [DR_NO_MARL]", file = file)
            
    
        """ ATE estimation
        """
        if i == 0:
            V_0 = V_OPE #  rep * n_estimates
            MC_0 = V_MC # a number
        else:  # 【bias, variance, MSE】 for 【\hat{V_target} - \hat{V}_0】
            ATE = arr(V_OPE) - V_0
            MC_ATE = V_MC - MC_0
            printG("MC-based ATE = " + str(round(MC_ATE, 2)))
            
            bias = np.round(np.abs(np.mean(ATE, 0) - MC_ATE), 2)
            std = np.round(np.std(ATE, 0), 2)
            mse = np.round(np.sqrt(bias**2 + std**2), 2)
            mse_rel = np.round(mse - mse[0], 2)
            bias = list(bias); std = list(std); mse = list(mse); mse_rel = list(mse_rel) 
            res = "   [DR/QV/IS]; [DR/QV/IS]_NO_MARL; [DR2]" + "\n" + "bias:" + str([bias[:3]]) + str([bias[3:6]]) + str([bias[6:9]]) + str([bias[9]]) + "\n" + "std:" + str([std[:3]]) + str([std[3:6]]) + str([std[6:9]]) + str([std[9]]) + "\n" + \
                    "MSE:" + str([mse[:3]]) + str([mse[3:6]]) + str([mse[6:9]]) + str([mse[9]])
            print(res)
            printB("MSE(-DR):" + str([mse_rel[:3]]) + str([mse_rel[3:6]]) + str([mse_rel[6:9]]) + str([mse_rel[9]])) #  + "\n"
            if file is not None:        
                print(res + "\n" + "MSE(-DR):" + str([mse_rel[:3]]) + str([mse_rel[3:6]]) + str([mse_rel[6:9]]) + str([mse_rel[9]]) + "\n", file = file)

            if mse_rel[0] <= np.min(mse_rel[2:4]):
                cprint("***** BETTER THAN [IS, DR_NO_MARL] *****", 'white', 'on_red')
                print(" ***** BETTER THAN [IS, DR_NO_MARL] *****", file = file)
                good_setting_flag += 1
            elif mse_rel[0] <= np.min(mse_rel[3:6]):     
                cprint("better than DR_NO_MARL", "yellow")
                print("better than [DR_NO_MARL]", file = file) 
#         print("--------------------- \n")
        cprint('==============',  attrs = ["bold"])
    if good_setting_flag == int(len(target_policys) * 2):
        printR("******************** THIS SETTING IS GOOD ********************")
    return None#[V_OPE, V_MC, std_V_MC]


def simu_once(seed = 1, l = 3, T = 14 * 24, time_dependent = False, 
              target_policys = None, 
              sd_D = 3, sd_R = 0, sd_O = 1, 
              CV_QV = False, 
              u_O = None, 
              dim_S_plus_Ts = 3 + 3, n_cores = n_cores, w_A = 1, w_O = .05, 
              penalty = [1, 1], penalty_NMF = None, 
              n_layer = 2, 
              w_hidden = 10, Learning_rate = 1e-4,  
              batch_size = 64, max_iteration = 1001, epsilon = 1e-3, 
               inner_parallel = False): 
    npseed(seed)
    N = l ** 2
    def behav(s, a):
        return 0.5
    behav = list(itertools.repeat(behav, N))
    
    def Ts(S_neigh):
        return np.mean(S_neigh, 0)
    def Ta(A_neigh):
        return Ta_disc(np.mean(A_neigh, 0), simple = simple)
    
    # observed data following behav
    data, adj_mat, details = DG_once(seed = seed, l = l, T = T, 
                                     u_O = u_O, 
                                     time_dependent = time_dependent,  
                                     sd_D = sd_D, sd_R = sd_R, sd_O = sd_O, 
                                     w_A = w_A, w_O = w_O)
    
    # OPE
    value_targets = []
    count = 1
    for target_policy in target_policys:
        value_estimators = V_DR(data = data, pi = target_policy, behav = behav, 
                                adj_mat = adj_mat, dim_S_plus_Ts = dim_S_plus_Ts, n_cores = n_cores, 
                             time_dependent = time_dependent,  
                             Ts = Ts, Ta = Ta, penalty = penalty, penalty_NMF = penalty_NMF, 
                                n_layer = n_layer, CV_QV = CV_QV, 
                            w_hidden = w_hidden, Learning_rate = Learning_rate,  
                            batch_size = batch_size, max_iteration = max_iteration, epsilon = epsilon, 
                             inner_parallel = inner_parallel)
        value_targets.append(value_estimators)
        print(count, end = " "); count += 1
    return value_targets

##########################################################################################################################################################
""" Apply MC to get the target value
"""
def MC_Value(l, T, time_dependent, target_policy, u_O = None, 
             sd_D =  3, sd_R = 0, sd_O = 1, 
             reps = 100, inner_parallel = False, w_A = 1, w_O = .01):
    def oneTraj(seed):
        # Off-policy evaluation with target_policy
        data, adj_mat, details = DG_once(seed = seed, l = l, T = T, time_dependent = time_dependent, 
                                          u_O = u_O, sd_O = sd_O, 
                                         TARGET = True, target_policy = target_policy, 
                                         sd_D = sd_D, sd_R = sd_R, w_A = w_A, w_O = w_O)
        R = details[2]
        V = np.mean(R)
        return V
    if inner_parallel:
        Vs = parmap(oneTraj, range(reps), n_cores)
    else:
        Vs = rep_seeds(oneTraj, reps)
    # std among reps is small
    res = np.round([np.mean(Vs), np.std(Vs)], 3)  
    printG("MC-based mean and std of average reward:" + str(res))
    return res
