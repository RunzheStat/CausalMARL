from _uti_basic import *
from _utility import *
from weight import *

def V_DR(data, pi, behav, adj_mat, Ts, Ta, penalty, dim_S_plus_Ts, n_cores = 10,
            time_dependent = False, 
            w_hidden = 10, Learning_rate = 1e-4,  n_layer = 2, CV_QV = False, 
            batch_size = 64, max_iteration = 1001, test_num = 0, epsilon = 1e-3, inner_parallel = False,
            competing_methods = True, print_flag = False): 
    """ Compute the average reward estimate of DR (and other 4 methods) with one dataset
    Input: 
        data: a len-N list. data[i] is a len-T list, where data[i][t] is [S_{i,t}, A_{i,t}, R_{i,t}]; 
            for now, we only use the state in data[i][T]
        policies:
            pi: a list (len-N) of the target policies
            behav: a list (len-N) of the behaviour policy
            pi(s, a) = a probability # although in didi, may not be the case
        adj_mat: binary adjacent matrix
        Ts, Ta: required spatial dependence functions
        competing_methods: if calculate the values with the other four methods
    Output: the average cumulative reward
    """ 
    N, T = len(data), len(data[0]) - 1
    Qi_diffs, V, w_all, values = [], [], [], []
    R = np.mean(np.array([[at[2] for at in a] for a in data]).T, 1)[:T]
    neigh = adj2neigh(adj_mat) # a dictionary, where neigh[i] is the list of region indeies for i's neighborhoood
    if time_dependent:
        dim_S_plus_Ts += 1
    def getOneRegionValue(i):
        a = now()
        """ get data """
        data_neigh = [[j, data[j]] for j in neigh[i]] # for region j, is a (len-#neigh) list of [index, data_at_that_index]
        Ri = arr([a[2] for a in data[i]])[:T]
        No_IS_V = np.mean(Ri)
        return No_IS_V

    if inner_parallel:
        r = arr(parmap(getOneRegionValue, range(N), n_cores))
    else:
        r = arr([getOneRegionValue(i) for i in range(N)])
    
        
    Vs = np.round(np.mean(r, 0), 3)
    return Vs

def getRegionData(data_i, i, data_neigh, pi, Ts, Ta, mean_field = True, time_dependent = False):
    """ Transform the data into transition tuples and extract spatial dependence statistics. 
    Output:
        tuples_i: a list of transition tuples. 
            tuples_i[t] = [S_it, A_it, R_it, Tsit, Tait, S_i(t+1), Tsi(t+1), pi_Sit_1, T_ait_1_pi]
    """
    T = len(data_i) - 1
    tuples_i = []
    for t in range(T):
        tuple_t = data_i[t].copy()
        if mean_field:
            Tsit = Ts([a[1][t][0] for a in data_neigh]) # a list (len-#neigh) of state at time t
            Tait = Ta([a[1][t][1] for a in data_neigh])
            S_it1 = data_i[t + 1][0]
            A_it1 = data_i[t + 1][1]
            S1_neigh = [a[1][t + 1][0] for a in data_neigh]
            Tsit1 = arr(Ts(S1_neigh))
            pi_Sit_1 = pi[i](S_it1, random_choose = True)
            T_ait_1_pi = Ta([pi[a[0]](a[1][t + 1][0], random_choose = True) for a in data_neigh])
        else:
            Tsit = np.vstack([a[1][t][0] for a in data_neigh]) # a list (len-#neigh) of state at time t
            Tait = np.vstack([a[1][t][1] for a in data_neigh])
            S_it1 = data_i[t + 1][0]
            A_it1 = data_i[t + 1][1]
            S1_neigh = [a[1][t + 1][0] for a in data_neigh]
            Tsit1 = np.vstack(S1_neigh)
            pi_Sit_1 = pi[i](S_it1, random_choose = True)
            T_ait_1_pi = np.vstack([pi[a[0]](a[1][t + 1][0], random_choose = True) for a in data_neigh])
        if time_dependent:
            tuple_t[0] = np.append(tuple_t[0], t % 48) # not general enough. specific to our applications.            
            S_it1 = np.append(S_it1, t % 48)
        tuple_t += [Tsit, Tait, S_it1, Tsit1, pi_Sit_1, T_ait_1_pi, A_it1]
        tuples_i.append(tuple_t)
    return tuples_i

def getWeight(tuples_i, policy0, policy1, dim_S_plus_Ts, time_dependent = False, 
              w_hidden = 10, Learning_rate = 1e-4,  n_layer = 2, 
              batch_size = 64, max_iteration = 1001, test_num = 0, epsilon = 1e-3,
             spatial = True):
    """ Compute the transition tuple density ratios for region i. [Breaking, Lihong]
    Input:  
        tuples_i[t] = [S_it, A_it, R_it, Tsit, Tait,  # 0 - 4
                        S_i(t+1), Tsi(t+1), pi_Sit_1, T_ait_1_pi] # 5 - 8
        What we want: SASR_i = [a list of [S,A,S',R]]; R is useless 
    Output: a vector of density ratios.
    """
    # prepare transition pairs
    # why we need action here?
    reg_weight = 0 # no penalty in the two-layer NN
    if spatial:
        def concateOne(tuplet):
            return [np.concatenate((tuplet[0], tuplet[3]), axis=None), 
                   tuplet[1],
                    np.concatenate((tuplet[5], tuplet[6]), axis=None), 
                   tuplet[2]]
    else:
        def concateOne(tuplet):
            return [tuplet[0], tuplet[1],
                    tuplet[5], tuplet[2]]
    SASR_i = [concateOne(tuplet) for tuplet in tuples_i]
    
    SASR_i = [SASR_i] # although we only need 1 layer of list
    
    if spatial:
        computeWeight = Density_Ratio_kernel(obs_dim = dim_S_plus_Ts, n_layer = n_layer, 
                                     w_hidden = w_hidden, Learning_rate = Learning_rate, reg_weight = reg_weight)
    else:
        if time_dependent: # general?
            obs_dim = int(dim_S_plus_Ts / 2) + 1
        else:
            obs_dim = int(dim_S_plus_Ts / 2)
        computeWeight = Density_Ratio_kernel(obs_dim = obs_dim, n_layer = n_layer, 
                                     w_hidden = w_hidden, Learning_rate = Learning_rate, reg_weight = reg_weight)
    
    weights = computeWeight.train(SASR_i, policy0, policy1, 
              batch_size = batch_size, max_iteration = max_iteration, 
              test_num = test_num, epsilon = epsilon)
    computeWeight.close_Session()
    
    return weights




def computeQV(tuples_i, R, dim_S_plus_Ts, spatial = True, CV_QV = False,
              penalty_range = [[0.01], [.01]], K_CV = 3, print_flag = False):
    """ Value functions [Susan Murphy] w/o CV
    Input:
        tuples_i: required data for the region; 
            tuples_i[t] = [S_it, A_it, R_it, Ts_it, Ta_it, # 0 - 4
                            S_i(t+1), Tsi(t+1), pi_Sit_1, T_ait_1_pi] # 5 - 8
        pi: the target policy
        
    Output:
        Vi: \hat{V}_{i,pi}
        Qi_diff: a vector (len-T) of Q^pi(pi) - Q^pi(behav)
    """
    if CV_QV is False:
        penalty = [a[0] for a in penalty_range]
        return computeQV_basic(tuples_i = tuples_i, R = R, penalty = penalty, dim_S_plus_Ts = dim_S_plus_Ts, 
                        spatial = spatial, print_flag = print_flag)
    else:
        """ randomness?
        """
        kf = KFold(n_splits = K_CV) # shuffle = False -> no randomness
        min_Bellman_error = 1e10
        optimal_penalty = [None, None]
        for mu in penalty_range[0]:
            for lam in penalty_range[1]:
                Bellman_error = 0 
                for train_index, valid_index in kf.split(tuples_i):
                    train_tuples = [tuples_i[i] for i in train_index]
                    valid_tuples = [tuples_i[i] for i in valid_index]
                    Bellman_error += computeQV_basic(tuples_i = train_tuples, R = arr([R[i] for i in train_index]), penalty = [mu, lam], 
                                    dim_S_plus_Ts = dim_S_plus_Ts, spatial = spatial, 
                                    validation_set = valid_tuples)
#                     validation_set = [valid_tuples, arr([R[i] for i in valid_tuples])]
                if Bellman_error < min_Bellman_error:
                    min_Bellman_error = Bellman_error
                    optimal_penalty = [mu, lam]
        return computeQV_basic(tuples_i = tuples_i, R = R, penalty = optimal_penalty, 
                               dim_S_plus_Ts = dim_S_plus_Ts, spatial = spatial)
                    



def computeQV_basic(tuples_i, R, penalty, dim_S_plus_Ts, spatial = True, 
                    validation_set = None, print_flag = False):
    """ Value functions [Susan Murphy] w/o CV
    Input:
        tuples_i: required data for the region; 
            tuples_i[t] = [S_it, A_it, R_it, Ts_it, Ta_it, # 0 - 4
                            S_i(t+1), Tsi(t+1), pi_Sit_1, T_ait_1_pi] # 5 - 8
        pi: the target policy
        
    Output:
        Vi: \hat{V}_{i,pi}
        Qi_diff: a vector (len-T) of Q^pi(pi) - Q^pi(behav)
    """
    
    R = arr([a[2] for a in tuples_i]) # 0309 I begins to use Rit. maybe this is where the bias come from
    
    T = len(tuples_i)
    mu, lam = penalty
    if spatial:
        Z = np.array([np.concatenate((a[0], [a[1]], a[3], [a[4]])) for a in tuples_i]) # T * p. [S, A, Ts, Ta]
        Zstar = np.array([np.concatenate((a[5], [a[7]], a[6], [a[8]])) for a in tuples_i])
    else:
        Z = np.array([np.concatenate((a[0], [a[1]])) for a in tuples_i]) # T * p. [S, A, Ts, Ta]
        Zstar = np.array([np.concatenate((a[5], [a[7]])) for a in tuples_i])
    Z_tilde = np.vstack((Z, Zstar))
    A_set = set([a[1] for a in tuples_i])
    
    gamma_g = 1 / (2 * np.median(pdist(Z))**2); gamma_q = 1 / (2 * np.median(pdist(Z_tilde))**2)

    Kg = GRBF(Z, Z, gamma_g) + identity(T) * 1e-8
    
    KQ = GRBF(Z_tilde, Z_tilde, gamma_q) + identity(2 * T) * 1e-8
    """ centeralization, p11
    """
    ZTstar = np.mean(Z_tilde, 0)
    KQ = KQ - GRBF(Z_tilde, ZTstar.reshape(1, -1), gamma_q) - GRBF(ZTstar.reshape(1, -1), Z_tilde, gamma_q) - GRBF(ZTstar.reshape(1, -1), ZTstar.reshape(1, -1), gamma_q)[0][0]
    
    
    C = np.hstack((-identity(T),identity(T)))        
    vec1, I = ones(T), identity(T)
    E_right_bef_inverse = Kg + T * mu * I 
    
    CKQ_1 = np.hstack((C.dot(KQ), np.expand_dims(-vec1, 1)))
    a = now()
    ECKQ1 = Kg.T.dot(solve(E_right_bef_inverse, CKQ_1))
    if print_flag:
        print("solve1 in QV time cost:", now() - a, E_right_bef_inverse.shape, CKQ_1.shape)
    left = (ECKQ1.T.dot(ECKQ1) + np.vstack((np.hstack((T * lam * KQ, zeros((2 * T, 1)))), zeros((1, 2 * T + 1)))))
    
    a = now()
    right = ECKQ1.T.dot(Kg.dot(solve(E_right_bef_inverse, R)))
    alpha_eta = -solve(left, np.expand_dims(right,1)) 
    alpha = alpha_eta[:(len(alpha_eta) - 1)]
    Vi = eta = alpha_eta[-1]
    
    """ no centeralization for KQ
    """
    if validation_set is not None:
        R = arr([a[2] for a in validation_set]) # dim?
        SA_t = np.array([np.concatenate((a[0], [a[1]], a[3], [a[4]])) for a in validation_set])
        T = SA_t.shape[0]
        QSA = alpha.T.dot(GRBF(Z_tilde, SA_t, gamma_q) + identity(2 * T)* 1e-8).T
        QSA1 = 0 * QSA
        for action in A_set:
            """wait. incorrect
            CV wait. use fixed parameters first
            """
            SA_t1 = arr([[a[5], action] for a in validation_set])
            QSA1 += alpha.T.dot(GRBF(Z_tilde, SA_t, gamma_q) + identity(2 * T)* 1e-8).T
#             print(SA_t1.shape, alpha.shape, SA_t.shape, T, QSA1)
        bellman_errors = R + QSA1 - eta - QSA
        kernel = DotProduct() + WhiteKernel()
        gpr = GaussianProcessRegressor(kernel=kernel,
            random_state=0).fit(SA_t, bellman_errors)
        return np.mean(gpr.predict(SA_t)**2)  
        
    else:
        Qvalues = alpha.T.dot(KQ)
        Qi_diff = Qvalues[0, T:] - Qvalues[0, :T]
        return Qi_diff, Vi

    #- GRBF(Z, Zstar.reshape(1, -1), gamma_g) - GRBF(Zstar.reshape(1, -1), Z, gamma_g)  - GRBF(Zstar.reshape(1, -1), Zstar.reshape(1, -1), gamma_g)[0][0]
    
