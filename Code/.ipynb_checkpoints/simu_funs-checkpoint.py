from _uti_basic import *
from _utility import *
from weight import *
from main import *
from simu_funs import *
from simu_DGP import *
##########################################################################################################################################################

""" apply MC to sample long trajectoy and use kernel to learn two density functions. 
apply this MC-based function to approaximate the density ratio in observed data
bandwitch selected by CV to be 1
"""
def MC_validate_Weight_QV(l, u_O, time_dependent, sd_D, sd_R, 
                          w_A, w_O, neigh, target_policy, rep = 10, dynamics = "old"):
    # tuples_i[t] = [S_it, A_it, R_it, Tsit, Tait, S_i(t+1), Tsi(t+1), pi_Sit_1, T_ait_1_pi]
    # data: a len-N list. data[i] is a len-T list, where data[i][t] is [S_{i,t}, A_{i,t}, R_{i,t}]; 
    from sklearn.ensemble import RandomForestRegressor
    den_funs = []
    bandwidth_range = {'bandwidth': [0.1, 1]}
    def once(seed):
        T = 10000
        data, adj_mat, details = DG_once(seed = seed, l = l, T = T, 
                                         u_O = u_O, dynamics = dynamics, 
                                         time_dependent = time_dependent,  
                                         sd_D = sd_D, sd_R = sd_R, 
                                         w_A = w_A, w_O = w_O)

        data_target, adj_mat, details = DG_once(seed = seed, l = l, T = T, 
                                         u_O = u_O, dynamics = dynamics, 
                                         time_dependent = time_dependent,  
                                         TARGET = True, target_policy = target_policy, 
                                         sd_D = sd_D, sd_R = sd_R, 
                                         w_A = w_A, w_O = w_O)
        def temp(a, random_choose = True):
            return 1
        pi = [temp for i in range(100)]
        def Ts(S_neigh):
            return np.mean(S_neigh, 0)
        def Ta(A_neigh):
            return Ta_disc(np.mean(A_neigh, 0))
        den_fun = []
        Q_func = []
        T -= 1
        for i in range(l**2): 

            data_neigh = [[j, data[j]] for j in neigh[i]] # for region j, is a (len-#neigh) list of [index, data_at_that_index]
            tuples_i = getRegionData(data[i], i, data_neigh, pi, Ts, Ta, mean_field = True, time_dependent = time_dependent)
            behav_states = arr([ np.append(tuples_i[t][0], tuples_i[t][3]) for t in range(T)])

            data_neigh = [[j, data_target[j]] for j in neigh[i]] # for region j, is a (len-#neigh) list of [index, data_at_that_index]
            tuples_i = getRegionData(data_target[i], i, data_neigh, pi, Ts, Ta, mean_field = True, time_dependent = time_dependent)
            target_states = arr([np.append(tuples_i[t][0], tuples_i[t][3]) for t in range(T)])

            # normalization for kernel
            std_behav_states = np.std(behav_states, 0)
            std_target_states = np.std(target_states, 0)
            
            den_fun_b =  KernelDensity(kernel='gaussian', bandwidth= 1).fit(behav_states / std_behav_states)
            den_fun_target = KernelDensity(kernel='gaussian', bandwidth = 1).fit(target_states / std_target_states)
            den_fun.append([den_fun_b, den_fun_target, std_behav_states, std_target_states])
            
            
            R = [SAR[2] for SAR in data_target[i]]
            
            Q_data = []
            SA = np.array([np.concatenate((a[0], [a[1]], a[3], [a[4]])) for a in tuples_i])
            
            drop = 5000 - 1
            V = np.mean(R[(drop + 1):])
            SA = SA[:(len(SA) - drop)]
            V_t = []
            for t in range(len(data_target[i]) - drop - 1):
#                 V_t.append(np.sum(R[t:] - V))
                V_t.append(np.sum(R[t:(drop + 1)] - V)) # np.mean(R[t:])
            V_t = arr(V_t).reshape(-1,1)
            Q_func.append(RandomForestRegressor(max_depth = 10, random_state=0).fit(SA, V_t))
        print(seed, "DONE!")
        return [den_fun, Q_func]
    
    den_funs_Q_funcs = parmap(once, range(rep))
    def MC_den_ratio_fun(sample, sample_SA, l): 
        den_ratios = []
        Qs = []
        for i in range(rep):
            den_fun = den_funs_Q_funcs[i][0]
            # normalization for kernel
            pi_den = np.exp(den_fun[l][1].score_samples(sample / den_fun[l][3]))
            behav_den = np.exp(den_fun[l][0].score_samples(sample/ den_fun[l][2]))
            # calculate the ratios
            den_ratio = pi_den / behav_den
            den_ratio /= np.mean(den_ratio)
            den_ratios.append(den_ratio)
            Qs.append(den_funs_Q_funcs[i][1][l].predict(sample_SA[1]) - den_funs_Q_funcs[i][1][l].predict(sample_SA[0]))
        den_ratios = arr(den_ratios)
        Qs = arr(Qs)
        print("std among MC reps:", np.median(np.std(den_ratios,0)), np.median(np.std(den_ratios,1)))
        return  np.mean(den_ratios, 0), np.mean(Qs, 0) # pi / behav
    return MC_den_ratio_fun

##########################################################################################################################################################

def simu(pattern_seed = 1,  l = 3, T = 14 * 24, dynamics = "old", # Setting - general
         OPE_rep_times = 20, n_cores = n_cores, inner_parallel = False,  # Parallel
         time_dependent = False, # DGP / target
         dim_S_plus_Ts = 3 + 3, epsilon = 1e-6,  # fixed
         sd_D = 3, sd_R = 0, # noises
         w_A = 1, w_O = .05, # Setting - spatial
         penalty = [.01, .01], CV_QV = False, # QV parameters
         n_layer = 2, w_hidden = 30, # NN structure
         Learning_rate = 1e-4, batch_size = 32, max_iteration = 1001, # NN training
         test_num = 0, isValidation = False,  # debug
         file = None # echo
        ): 
    
    printR(str(EST()) + "; num of cores:" + str(n_cores) + "\n")
#     print("mean, std of MC;", "mean, std of wi;", "mean/median of (MC-wi);", "R2")
    # target_policy = simu_target_policy_pattern(pattern_seed, l, random = random_target)

    """ generate the order pattern (used by all rep, shared by b and pi)
    """
    if pattern_seed is None: # fixed
        u_O = [[6 for j in range(5)] for i in range(5)]
        u_O[1][1] = u_O[1][3] = u_O[3][1] = u_O[3][3] = 18
        u_O[2][2] = 24
        a = u_O[0]
        for i in range(1,5):
            a += u_O[i]
        u_O = a
    else: # randomly from logN
        # mean = 12.5, std = 3
        npseed(pattern_seed)
        u_O = rlogN(2.5, .2, l**2) 
        
    # generate the corresponding target plicy
    target_policy = simu_target_policy_pattern(u_O = u_O)
    # generate the adj for the grid
    neigh = adj2neigh(getAdjGrid(l))
    
    """ apply MC to validate QV / weight estimation
    """
    den_fun = None
    if isValidation:
        den_fun = MC_validate_Weight_QV(l = l, u_O = u_O, time_dependent = time_dependent, dynamics = dynamics,  
                                        sd_D = sd_D, sd_R = sd_R, w_A = w_A, w_O = w_O, 
                                        neigh = neigh, target_policy = target_policy)        
    
    """ parallel
    """
    def once(seed):
        return simu_once(seed = seed, l = l, T = T, time_dependent = time_dependent,  
                         u_O = u_O, den_fun = den_fun, dynamics = dynamics, 
                         target_policy = target_policy, w_A = w_A, w_O = w_O, dim_S_plus_Ts = dim_S_plus_Ts, n_cores = n_cores, 
                          penalty = penalty, n_layer = n_layer, sd_D = sd_D, sd_R = sd_R, 
                          w_hidden = w_hidden, Learning_rate = Learning_rate,  CV_QV = CV_QV, 
                          batch_size = batch_size, max_iteration = max_iteration, epsilon = epsilon, 
                          test_num = test_num, inner_parallel = inner_parallel, 
                          isValidation = isValidation)
    if not inner_parallel:
        V_OPE = parmap(once, range(OPE_rep_times), n_cores)
    else:
        V_OPE = rep_seeds(once, OPE_rep_times)
        
    """ MC-based average reward following the target
    """
    V_MC, std_V_MC = MC_Value(l = l, T = T, time_dependent = time_dependent, dynamics = dynamics,  
                              u_O = u_O, sd_D = sd_D, sd_R = sd_R, w_A = w_A, w_O = w_O, 
                              target_policy = target_policy, reps = 100, 
                              inner_parallel = inner_parallel)
    V_behav = np.mean(V_OPE, 0)[-1]
    print("Value of Behaviour policy:", round(V_behav, 3))
    bias = np.round(np.abs(np.mean(V_OPE, 0) - V_MC), 2)
#     bias = np.round(np.mean(V_OPE, 0) - V_MC, 2)
    std = np.round(np.std(V_OPE, 0), 2)
    mse = np.round(np.sqrt(bias**2 + std**2), 2)
    mse_rel = np.round(mse - mse[0], 2)
    bias = list(bias); std = list(std); mse = list(mse); mse_rel = list(mse_rel) # DR2,
    res = "   [DR/QV/IS]; [DR/QV/IS]_NO_MARL; [V_behav]" + "\n" + "bias:" + str([bias[:3]]) + str([bias[3:6]]) + str([bias[6:]]) + "\n" + "std:" + str([std[:3]]) + str([std[3:6]]) + str([std[6:]]) + "\n" + \
            "MSE:" + str([mse[:3]]) + str([mse[3:6]]) + str([mse[6:]]) + "\n" + \
            "MSE(-DR):" + str([mse_rel[:3]]) + str([mse_rel[3:6]]) + str([mse_rel[6:]]) + "\n"
    print(res)
    if mse_rel[0] <= np.min(mse_rel):
        printR("GOOD JOB!")
        
    if file is not None:        
        print(res, file = file)
    return [V_OPE, V_MC, std_V_MC]


def simu_once(seed = 1, l = 3, T = 14 * 24, time_dependent = False, dynamics = "old", 
              target_policy = None, 
              sd_D = 3, sd_R = 0, CV_QV = False, 
              u_O = None, den_fun = None,
              dim_S_plus_Ts = 3 + 3, n_cores = n_cores, w_A = 1, w_O = .05, 
              penalty = [1, 1], n_layer = 2, 
              w_hidden = 10, Learning_rate = 1e-4,  
              batch_size = 64, max_iteration = 1001, epsilon = 1e-3, 
              test_num = 0, inner_parallel = False,
             isValidation = False): 
    npseed(seed)
    N = l ** 2
    def behav(s, a):
        return 0.5
    behav = list(itertools.repeat(behav, N))
    
    def Ts(S_neigh):
        return np.mean(S_neigh, 0)
    def Ta(A_neigh):
        return Ta_disc(np.mean(A_neigh, 0))
    
    # observed data following behav
    data, adj_mat, details = DG_once(seed = seed, l = l, T = T, 
                                     u_O = u_O, dynamics = dynamics, 
                                     time_dependent = time_dependent,  
                                     sd_D = sd_D, sd_R = sd_R, 
                                     w_A = w_A, w_O = w_O)
    
    # OPE
    r = V_DR(data = data, pi = target_policy, behav = behav, adj_mat = adj_mat, dim_S_plus_Ts = dim_S_plus_Ts, n_cores = n_cores, 
             time_dependent = time_dependent, den_fun = den_fun, 
             Ts = Ts, Ta = Ta, penalty = penalty, n_layer = n_layer, CV_QV = CV_QV, 
            w_hidden = w_hidden, Learning_rate = Learning_rate,  
            batch_size = batch_size, max_iteration = max_iteration, epsilon = epsilon, 
            test_num = test_num, inner_parallel = inner_parallel,
            isValidation = isValidation)
    return r

##########################################################################################################################################################
""" Apply MC to get the target value
"""
def MC_Value(l, T, time_dependent, target_policy, u_O = None, dynamics = "old", 
             sd_D =  3, sd_R = 0, reps = 100, inner_parallel = False, w_A = 1, w_O = .01):
    def oneTraj(seed):
        # Off-policy evaluation with target_policy
        data, adj_mat, details = DG_once(seed = seed, l = l, T = T, time_dependent = time_dependent, 
                                          u_O = u_O, dynamics = dynamics, 
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
    print("MC-based mean and std of average reward:", res)
    return res

##########################################################################################################################################################
#     else:
#         # use either the target policy or its noisy version
#         behav_policy = simu_target_policy_pattern(pattern_seed = seed, l = l, random = True, u_O = u_O, 
#                                                   threshold = 10, noise = True)
#         data, adj_mat, details = DG_once(seed = seed, l = l, T = T, time_dependent = time_dependent, 
#                                           u_O = u_O, 
#                                          TARGET = True, target_policy = behav_policy, sd_D = sd_D,
#                                         sd_R = sd_R, w_A = w_A, w_O = w_O)
