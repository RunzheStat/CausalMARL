from utils import *

##########################################################################################################################################################
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
def DG_once(seed = 1, l = 5, T = 14 * 48, t_func = None, u_D = 100, 
            w_A = 1, w_O = 1, sd_R  = 1, sd_D = 1, sd_O = 1, 
            u_O = None, 
           TARGET = False, target_policy = None, T_burn_in = 100):  
    """ prepare data (fixed)
    """
    
    T = T + T_burn_in
    npseed(seed)
    N = l ** 2
    adj_mat = getAdjGrid(l)
    p_behav = 0.5

    e_R = randn(N, T) * sd_R
    # initialization
    M = [runi(0, 1, N)] 
    R = []
    w_M = 0.5 

    # O: the pattern (spatial distribution) of orders
    O = rpoisson(u_O, (T, N)).T    

    
    # Actions
    if TARGET: # target. fixed. 
        A = arr([[target_policy[i](None, random_choose = True) for j in range(T)] for i in range(N)])
    else: # behaviour policy: random
        A = rbin(1, p_behav, (N, T))
        
    D = [arr([u_D for i in range(N)])]
    
    """ MAIN: state trasition and reward calculation [no action selection]
    """
    n_neigh = np.sum(adj_mat,1)
    for t in range(1, T): 
        """ Drivers
        """
        ## attractions
        Attr_OD = w_O * (squeeze(O[:, t - 1]) / (squeeze(D[t - 1])))
        Attr = np.exp(w_A * A[:, t - 1]) + squeeze(Attr_OD) 
        
        Attr_mat = np.repeat(Attr, N).reshape(N, N)
        Attr_adj = np.multiply(Attr_mat, adj_mat)
        Attr_neigh = np.sum(Attr_adj, 0)

        D_t = squeeze(Attr_adj.dot((D[t - 1] / squeeze(Attr_neigh)).reshape(-1, 1)))

        D.append(D_t)
        O_t = O[:, t] 
        
        M_t = w_M * (1 - abs(D_t - O_t) / abs(1 + D_t + O_t)) + (1 - w_M) * M[t - 1]
        R_t_1 = M_t * np.minimum(D_t, O_t) + e_R[:, t] 
        
        M.append(M_t)
        R.append(R_t_1)
    R.append(R_t_1) 

    """ organization
    """
    ## organization and burn-in; N * T
    R = arr(R).T[:, T_burn_in:]; D = arr(D).T[:, T_burn_in:]; M = arr(M).T[:, T_burn_in:]
    O = O[:, T_burn_in:]; A = A[:, T_burn_in:]
    
    ## reorganization
    data = []
    for i in range(N):
        data_i = []
        for t in range(T - T_burn_in):
            data_i.append([arr([O[i, t], D[i, t], M[i, t]]), A[i, t], R[i, t]])
        data.append(data_i)
    
    return data, adj_mat, [[O, D, M], A, R]

##########################################################################################################################################################

""" generate the target policy (fixed reward regions) randomly / based on u_O
"""
def simu_target_policy_pattern(l = 3, u_O = None, threshold = 12, print_flag = True):
    
    if threshold >=0: # generate target based on the order
        N = len(u_O)
        l = int(sqrt(N))
        fixed_policy = [int(u_O[i] > threshold) for i in range(N)]
    else: # randomly  generate the target policy
        npseed(abs(threshold))
        N = l**2
        fixed_policy = list(rbin(1, 0.5, N))
    
    ## Transform fixed_policy (0/1) to policy (pi)
    pi = []
    for reward in fixed_policy:
        def pi_i(s = None, a = 0, random_choose = False, reward = reward):
            if random_choose:
                return reward
            else:
                return int(a == reward)
        pi.append(pi_i)
        
    ## Draw plot
    if print_flag == "all":
        print("means of Order:", "\n")
        if u_O is not None:
            for i in range(l):
                for j in range(l):
                    print(round(u_O[i * l + j], 1), end = " ")
                print("\n")
    if print_flag in ["all", "policy_only"]:
        print("target policy:", "\n")
        for i in range(l):
            for j in range(l):
                if fixed_policy[i * l + j] == 1:
                    print("1", end = " ")
                else:
                    print("0", end = " ")
            print("\n")
    print("number of reward locations: ", sum(fixed_policy))
    return pi
