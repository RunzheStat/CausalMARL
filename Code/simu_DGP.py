from _uti_basic import *
from _utility import *

#############################################################################
"""
    1. behaviour policy: always random 0.5 reward (DG-level)
    2. target policy
        1. based on generated u_O (simu-level), get fixed reward positions. (simu-level)
    
    Output:
        data: a len-N list. data[i] is a len-T list, where data[i][t] is [S_{i,t}, A_{i,t}, R_{i,t}]; 
            for now, we only use the state in data[i][T]
        adj_mat: binary adjacent matrix
        [[O, D, M], A, R]
"""
def DG_once(seed = 1, l = 5, T = 240, time_dependent = False, w_A = 1, w_O = 1, sd_R  = 1, sd_D = 1, 
            u_O = 12, p_behav = 0.5, M_in_R = True, #True,
           OPE = False, target_policy = None, T_burn_in = 50, mean_reversion = False, dynamics = "new"):  
    """ prepare data
    """
    T = T + T_burn_in
    npseed(seed)
    N = l ** 2
    
    """ weight in the definition of mismatch
    """
    w_M = 0.8 
    adj_mat = getAdjGrid(l)
    
    """ O: the pattern (spatial distribution) of orders
    """
    is_homogenous = isinstance(u_O, int)
    if is_homogenous: # u_O is a number
        if time_dependent:
            mean_OD = [10 * (2 - sin(t/48*2*np.pi)) for t in range(T)]
            O = rpoisson(mean_OD, (N, T))
        else: # generate orders from poission distribution with E(O) = u_O
            O = rpoisson(u_O, (N, T))  
    else: # list, heterogenous 
        O = rpoisson(u_O, (T, N)).T
    
    """ debug: is the variaance of O too large? learn nothing?
    """
    O = np.repeat(u_O, T).reshape(N, T) + randn(N,T) #/ 10
    
    """ D: initial is the same with driver. then attract by the A and O. burn-in.
    """
    u_D = np.mean(u_O)
    D = [arr([u_D for i in range(N)])]
    
    """ Actions
    """
    if OPE: # target. fixed. 
        A = arr([[target_policy[i](None, random_choose = True) for j in range(T)] for i in range(N)])
    else: # behaviour policy: random
        A = rbin(1, p_behav, (N, T))
    
    """ random errors for D and R
    """
    e_D = (rpoisson(1, (N,T)) - 1) * sd_D
    e_R = randn(N, T) * sd_R

    """ initialization
    """
    M = [runi(0, 1, N)] 
    R = []
    
    """ MAIN: state trasition and reward calculation [no action selection]
    """
    n_neigh = np.sum(adj_mat,1)
    for t in range(1, T): 
        """ Drivers
        """
        ## attractions
        if dynamics == "old":
            D_t = diag(1 + w_O * O[:, t - 1]).dot(identity(N) + \
                                        w_A * diag(A[:, t - 1])).dot(adj_mat).dot(diag( 1 / (O[:, t - 1] * w_O + 1))).dot(D[t - 1])
            D_t = D_t / n_neigh

            D_t = np.round(arr([a for a in D_t]) + e_D[:, t])
            D_t[D_t < 0] = 0

            # normalization
            D_t = D_t * (sum(D[t - 1])/sum(D_t))
            D.append(D_t)
        else:
                
            Attr_OD = w_O * squeeze(O[:, t - 1]) / (1 + squeeze(D[t - 1]))
            Attr = np.exp(w_A * A[:, t - 1]) + squeeze(Attr_OD) ## * at first
            Attr_mat = np.repeat(Attr, N).reshape(N, N)
            Attr_adj = np.multiply(Attr_mat, adj_mat)
            Attr_neigh = np.sum(Attr_adj, 0)
            
            D_t = squeeze(Attr_adj.dot((D[t - 1] / squeeze(Attr_neigh)).reshape(-1, 1)))
            D.append(D_t)
        """ New Order and Mismatch
        """
        O_t = O[:, t]
        M_t = w_M * (1 - abs(D_t - O_t) / abs(D_t + O_t)) + (1 - w_M) * M[t - 1]
        M.append(M_t)
        
        """ Reward definitions
        """
        if M_in_R:
            R_t_1 = M_t * np.minimum(D_t, O_t) + e_R[:, t] # R_{t - 1}
        else:
            R_t_1 = np.minimum(D_t, O_t) + e_R[:, t]
        R.append(R_t_1)
    R.append(R_t_1) # add one more? # new reward?
    
    ## organization and burn-in; N * T
    R = arr(R).T[:, T_burn_in:]; D = arr(D).T[:, T_burn_in:]; M = arr(M).T[:, T_burn_in:]
    O = O[:, T_burn_in:]; A = A[:, T_burn_in:]
        
    """ reorganization
    """
    data = []
    for i in range(N):
        data_i = []
        for t in range(T - T_burn_in):
            data_i.append([arr([O[i, t], D[i, t], M[i, t]]), A[i, t], R[i, t]])
        data.append(data_i)
    
    return data, adj_mat, [[O, D, M], A, R]



""" generate the target policy (fixed reward regions) randomly / based on u_O
"""
def simu_target_policy_pattern(pattern_seed = 1, l = 3, random = True, u_O = None, threshold = 12, print_flag = True, noise = False):
    
    if u_O is not None: # generate target based on the order
        N = len(u_O)
        l = int(sqrt(N))
        fixed_policy = [int(u_O[i] > threshold) for i in range(N)]
    else: # randomly  generate the target policy
        npseed(pattern_seed)
        N = l**2
        if random:
            fixed_policy = rbin(1, 0.5, N)
        else:
            """ fixed number: half as reward
            """
            reward_place = np.random.choice(N, N//2, replace = False)
            fixed_policy = [int(b in reward_place) for b in range(N)]
    if noise:
        e = np.random.choice(N, int(N//5), replace = False)
        fixed_policy -= e
        fixed_policy = abs(fixed_policy)
    
    ## Transform fixed_policy (0/1) to policy (pi)
    pi = []
    for reward in fixed_policy:
        def pi_i(s, a = 0, random_choose = False, reward = reward):
            if random_choose:
                return reward
            else:
                return int(a == reward)
        pi.append(pi_i)
        
    ## Draw plot
    if print_flag:
        print("target policy:", "\n")
        for i in range(l):
            for j in range(l):
                if fixed_policy[i * l + j] == 1:
                    print("1", end = " ")
                else:
                    print("0", end = " ")
            print("\n")
        print("means of Order:", "\n")
        if u_O is not None:
            for i in range(l):
                for j in range(l):
                    print(round(u_O[i * l + j], 3), end = " ")
                print("\n")
    return pi



# mean reversion for stationality; 
            # then how about attraction? not ideal? action effect?
            # can be put before P(). Then no problem any longer.
#         if not mean_reversion:

#         else: # mean-reversion
#             if time_dependent:
#                 D_t = (D_t + mean_OD[t]) // 2  
#             else:
#                 if is_homogenous:
#                     D_t = (D_t + 10) // 2 
#                 else:
#                     D_t = (D_t + u_O) // 2 
#             D_t = np.round(arr([a for a in D_t]) + e_D[:, t])


# more spatial component (spatial reward); not exactly same with Chengchun's
# new reward definitions
#     neigh = adj2neigh(adj_mat)
#     R_spatial_more = []
#     for i in range(N):
#         R_neigh = []
#         for j in neigh[i]:
#             R_neigh.append(R[j,:])
#         R_spatial_more.append(R[i,:] + np.mean(R_neigh, 0))
#     R = arr(R_spatial_more)
