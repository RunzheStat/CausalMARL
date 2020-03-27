from _uti_basic import *
from _utility import *
from weight import *
from main import *
from simu_funs import *
#############################################################################

def simu_once(seed = 1, l = 3, T = 10, time_dependent = False, 
              target_policy = None, mean_reversion = False, 
              sd_D = 1, sd_R = 0, CV_QV = False, 
              print_flag_ = False, 
              u_O = 10, 
              dim_S_plus_Ts = 3 + 3, n_cores = n_cores, w_A = 1, w_O = 1, 
              penalty = [1, 1], n_layer = 2, simple_grid_neigh = True, 
              w_hidden = 10, Learning_rate = 1e-4,  
              batch_size = 64, max_iteration = 1001, epsilon = 1e-3, 
              test_num = 0, inner_parallel = False): 
    npseed(seed)
    N = l ** 2
    def behav(s, a):
        return 0.5
    behav = list(itertools.repeat(behav, N))
    
    def Ta_disc(ta):
        if ta <= 2/8:
            return 0
        if ta <= 5/8:
            return 1
        else:
            return 2
    
    def Ts(S_neigh):
        return np.mean(S_neigh, 0)
    def Ta(A_neigh):
        return Ta_disc(np.mean(A_neigh, 0))
    
    # DG
    data, adj_mat, details = DG_once(seed = seed, l = l, T = T, 
                                     u_O = u_O, 
                                     time_dependent = time_dependent, mean_reversion = mean_reversion, 
                                     sd_D = sd_D, sd_R = sd_R, 
                                     w_A = w_A, w_O = w_O, 
                                     simple_grid_neigh = simple_grid_neigh)
    
    # OPE
    r = V_DR(data = data, pi = target_policy, behav = behav, adj_mat = adj_mat, dim_S_plus_Ts = dim_S_plus_Ts, n_cores = n_cores, 
             time_dependent = time_dependent, 
             Ts = Ts, Ta = Ta, penalty = penalty, n_layer = n_layer, CV_QV = CV_QV, 
            w_hidden = w_hidden, Learning_rate = Learning_rate,  
            batch_size = batch_size, max_iteration = max_iteration, epsilon = epsilon, 
            test_num = test_num, inner_parallel = inner_parallel, print_flag = print_flag_)
    return r


def simu(pattern_seed = 1, OPE_rep_times = 20, l = 3, T = 10, time_dependent = False, 
         mean_reversion = True, 
         dim_S_plus_Ts = 3 + 3, n_cores = n_cores, 
         print_flag = False, 
         sd_D = 1, sd_R = 0, u_O = 10, 
         simple_grid_neigh = True, w_A = 1, w_O = 1, random_target = True,
              penalty = [1, 1], n_layer = 2, CV_QV = False, 
              w_hidden = 10, Learning_rate = 1e-4,  
              batch_size = 64, max_iteration = 1001, epsilon = 1e-3, 
              test_num = 0, inner_parallel = False): 
    
    printR(str(EST()) + "; num of cores:" + str(n_cores))
    
    # target_policy = simu_target_policy_pattern(pattern_seed, l, random = random_target)
    target_policy = simu_target_policy_pattern(u_O = u_O)
    
    def once(seed):
        if print_flag and seed == 0:
            print_flag_ = True
        else:
            print_flag_ = False
        return simu_once(seed = seed, l = l, T = T, time_dependent = time_dependent, 
                         mean_reversion = mean_reversion, 
                         print_flag_ = print_flag_, 
                         u_O = u_O, 
                         target_policy = target_policy, w_A = w_A, w_O = w_O, dim_S_plus_Ts = dim_S_plus_Ts, n_cores = n_cores, 
              penalty = penalty, n_layer = n_layer, sd_D = sd_D, sd_R = sd_R, simple_grid_neigh = simple_grid_neigh, 
              w_hidden = w_hidden, Learning_rate = Learning_rate,  
                         CV_QV = CV_QV, 
              batch_size = batch_size, max_iteration = max_iteration, epsilon = epsilon, 
              test_num = test_num, inner_parallel = inner_parallel)
    if not inner_parallel:
        V_OPE = parmap(once, range(OPE_rep_times), n_cores)
    else:
        V_OPE = rep_seeds(once, OPE_rep_times)
    V_MC, std_V_MC = MC_Value(l, T, time_dependent = time_dependent, mean_reversion = mean_reversion, 
                              u_O = u_O, 
                              target_policy = target_policy, reps = 100, sd_D = sd_D, sd_R = sd_R, w_A = w_A, w_O = w_O, 
                              inner_parallel = inner_parallel, 
                             simple_grid_neigh = simple_grid_neigh)
    
    bias = np.round(np.abs(np.mean(V_OPE, 0) - V_MC), 3)
    std = np.round(np.std(V_OPE, 0), 3)
    print("    DR,  DR2,  IS,  Susan,  DR_NS,  No_IS_V", "\n", 
          "bias:", bias, "\n", 
          "std:", std, "\n", 
          "MSE:", np.round(np.sqrt(bias**2 + std**2), 3), 
          )
    return [V_OPE, V_MC, std_V_MC]
#############################################################################

def DG_once(seed = 1, l = 5, T = 240, time_dependent = False, w_A = 1, w_O = 1, sd_R  = 1, sd_D = 1, u_O = 10, 
           OPE = False, target_policy = None, simple_grid_neigh = True, T_burn_in = 50, mean_reversion = True):
    """
    Output:
        data: a len-N list. data[i] is a len-T list, where data[i][t] is [S_{i,t}, A_{i,t}, R_{i,t}]; 
            for now, we only use the state in data[i][T]
        adj_mat: binary adjacent matrix
    """
    T = T + T_burn_in
    npseed(seed)
    N = l ** 2
    w_M, p_behav = 0.5, 0.5
    
    adj_mat = getAdjGrid(l, simple_grid_neigh)
    
    if isinstance(u_O, int):
        if time_dependent:
            mean_OD = [10 * (2 - sin(t/48*2*np.pi)) for t in range(T)]
            O = rpoisson(mean_OD, (N, T))
        else:
            O = rpoisson(u_O, (N, T))  # generate orders from poission distribution with Exp = u_D (i.e., same variance). 
    else:
        u_O = u_O # list, heterogenous 
    
    """ I can add some pattern to D and O. but the pattern needs to be shared between them.
    """
    O = rpoisson(u_O, (T, N)).T
    
#     D = [rpoisson(u_D, N)] 
#     D = [u_O]
    D = [arr([12 for i in range(N)])]
    
    if OPE:
        A = arr([[target_policy[i](1, random_choose = True) for j in range(T)] for i in range(N)])
    else:
        A = rbin(1, p_behav, (N, T))
    e_D = (rpoisson(1, (N,T)) - 1) * sd_D
    e_R = randn(N, T) * sd_R
    ## initialization
    
    M = [runi(0, 1, N)] 
    R = []
    ## state trasition and reward calculation [no action selection]
    for t in range(1, T): 
        D_t = diag(O[:, t - 1]).dot(w_A * identity(N) + \
                                    diag(A[:, t - 1])).dot(adj_mat).dot(diag( 1 / (O[:, t - 1] + w_O))).dot(D[t - 1])
        
        # normalizatiomn
        D_t = sum(D[t - 1]) / sum(D_t) * D_t
        """ mean reversion for stationality; 
        then how about attraction? not ideal? action effect?
        can be put before P(). Then no problem any longer.
        """
        if not mean_reversion:
            D_t = np.round(arr([a for a in D_t]) + e_D[:, t])
            D_t[D_t < 1] = 1 # o.w., negetive; represent no drives
        else:
            if time_dependent:
                D_t = (D_t + mean_OD[t]) // 2  
            else:
                D_t = (D_t + 10) // 2 
            D_t = np.round(arr([a for a in D_t]) + e_D[:, t])

        D.append(D_t)
        
        O_t = O[:, t]
        M_t = w_M * (1 - abs(D_t - O_t) / abs(1 + D_t + O_t)) + (1 - w_M) * M[t - 1]
        M.append(M_t)
        
        R_t_1 = M_t * np.minimum(D_t, O_t) + e_R[:, t]
        R.append(R_t_1)
    R.append(R_t_1)
    
    ## organization; N * T
    R = arr(R).T[:, T_burn_in:]; D = arr(D).T[:, T_burn_in:]; M = arr(M).T[:, T_burn_in:]
    
    """ more spatial component; not exactly same with Chengchun's
    new reward definitions
    """
#     neigh = adj2neigh(adj_mat)
#     R_spatial_more = []
#     for i in range(N):
#         R_neigh = []
#         for j in neigh[i]:
#             R_neigh.append(R[j,:])
#         R_spatial_more.append(R[i,:] + np.mean(R_neigh, 0))
#     R = arr(R_spatial_more)
    
    ## reorganization
    data = []
    for i in range(N):
        data_i = []
        for t in range(T - T_burn_in):
            data_i.append([arr([O[i, t], D[i, t], M[i, t]]), A[i, t], R[i, t]])
        data.append(data_i)
    
    return data, adj_mat, [[O, D, M], A, R]

def simu_target_policy_pattern(pattern_seed = 1, l = 3, random = True, u_O = None):
    
    if u_O is not None:
        N = len(u_O)
        l = int(sqrt(N))
        fixed_policy = [int(u_O[i] > 12) for i in range(N)]
    else:
        npseed(pattern_seed)
        N = l**2
        if random:
    #         fixed_policy = rbin(1, 0.5, N)
            fixed_policy = rbin(1, 0.4, N)
        else:
            """ fixed number: half as reward
            """
            reward_place = np.random.choice(N, N//2, replace = False)
            fixed_policy = [int(b in reward_place) for b in range(N)]
        
    ## Transform 0/1 to policy
    pi = []
    for reward in fixed_policy:
        def pi_i(s, a = 0, random_choose = False, reward = reward):
            if random_choose:
                return reward
            else:
                return int(a == reward)
        pi.append(pi_i)
    ## Print
    for i in range(l):
        for j in range(l):
            if fixed_policy[i * l + j] == 1:
                print("1", end = " ")
            else:
                print("0", end = " ")
        print("\n")
    return pi

def MC_Value(l, T, time_dependent, target_policy, mean_reversion = True, u_O = 10, 
             sd_D =  1, sd_R = 0, reps = 100, inner_parallel = False, simple_grid_neigh = True, w_A = 1, w_O = 1):
    def oneTraj(seed):
        data, adj_mat, details = DG_once(seed = seed, l = l, T = T, time_dependent = time_dependent, 
                                         mean_reversion = mean_reversion, u_O = u_O, 
                                         OPE = True, target_policy = target_policy, sd_D = sd_D,
                                        sd_R = sd_R, simple_grid_neigh = simple_grid_neigh, w_A = w_A, w_O = w_O)
        R = details[2]
        V = np.mean(R)
        return V
    if inner_parallel:
        Vs = parmap(oneTraj, range(reps), n_cores)
    else:
        Vs = rep_seeds(oneTraj, reps) # put the parallelization in the outside.
    r = np.round([np.mean(Vs), np.std(Vs)], 3)  #/ np.sqrt(reps)
    print("MC-based mean [average reward] and its std:", r)
    return r

#############################################################################
def getAdjGrid(l, simple = True):
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


    
