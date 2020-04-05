from _uti_basic import *
from _utility import *
from weight import *
from simu_funs import *
##########################################################################################################################################################
##########################################################################################################################################################

""" Compute the average reward estimate using DR (and other methods) with one dataset
Input: 
    data: a list of length N, where N is the number of regions. data[i] is a list of length T, where data[i][t] is [S_{i,t}, A_{i,t}, R_{i,t}] 
    adj_mat: N * N adjacent matrix
    policies:
        pi: a list (len-N) of target policies
        behav: a list (len-N) of  behaviour policy
        where
            policy(s, a) = a probability 
            policy(s, random_choose = true) = sample an action
    adj_mat: binary adjacent matrix
    Ts, Ta: required spatial dependence functions.
    penalty: a list of two ranges of penalty parameters for QV method
    penalty_NMF: penalty for NO_MeanField
    dim_S_plus_Ts: dim of state + dim of Ts
    w_hidden, Learning_rate, n_layer, batch_size, max_iteration, epsilon: parameters for NN used in IS-based estimates.
Output: 
    a vector of length #estimator flr the average cumulative reward
""" 
def V_DR(data, adj_mat, pi, behav, Ts, Ta, 
         penalty = [[1e-4, 1e-4]], penalty_NMF = None, 
         dim_S_plus_Ts = 3 + 3, 
            time_dependent = False, simple = False,
            w_hidden = 30, Learning_rate = 1e-3,  n_layer = 2, CV_QV = False, 
            batch_size = 32, max_iteration = 1001,  epsilon = 1e-6, inner_parallel = False): 

    N, T = len(data), len(data[0]) - 1
    Qi_diffs, V, w_all, values = [], [], [], []
    R = np.mean(np.array([[at[2] for at in a] for a in data]).T, 1)[:T]
    neigh = adj2neigh(adj_mat) # a dictionary, where neigh[i] is the list of region indeies for i's neighborhoood
    if time_dependent:
        dim_S_plus_Ts += 1
    
    a = now()
    def getOneRegionValue(i):
        """ get data """
        data_neigh = [[j, data[j]] for j in neigh[i]] # for region j, is a (len-N_j) list of [index, data_at_that_index]
        n_neigh = len(neigh[i])
        Ri = arr([a[2] for a in data[i]])[:T]
        V_behav = np.mean(Ri) # the average reward of behavious policy at location i (no difference?)
        tuples_i = getRegionData(data[i], i, data_neigh, pi, Ts, Ta, mean_field = True, time_dependent = time_dependent)
        
        
        Ta_i = Ta_disc(np.mean([pi[j](s = None, random_choose = True) for j in neigh[i]]), simple = simple)
        
        """ our method """
        Qi_diff, Vi = computeQV(tuples_i = tuples_i, R = Ri, n_neigh = n_neigh, 
                                CV_QV = CV_QV, penalty_range = penalty, spatial = True)
        r = getWeight(tuples_i, i, policy0 = behav[i], policy1 = pi[i], Ta_i = Ta_i, dim_S_plus_Ts = dim_S_plus_Ts,
                       time_dependent = time_dependent, n_neigh = n_neigh,
                      w_hidden = w_hidden, Learning_rate = Learning_rate,  n_layer = n_layer, 
                      batch_size = batch_size, max_iteration = max_iteration,  
                      epsilon = epsilon)
        
        wi = r[0]
        wi /= np.mean(wi)
        DR_V = wi * (Ri + Qi_diff)
        
        """ COMPETING METHODS
        1. with MARL 
        """
        IS_V = wi * Ri
        QV_V = Vi[0] 
        DR2_V = IS_V + QV_V - DR_V # the DR proposed by that paper
        
        """ 2. DR_NO_MARL
        """

        Qi_diff_NS, Vi_NS = computeQV(tuples_i = tuples_i, R = Ri, 
                            CV_QV = CV_QV, penalty_range = penalty, spatial = False)
        QV_NS = Vi_NS[0]
        
        wi_NS = getWeight(tuples_i, i, policy0 = behav[i], policy1 = pi[i], Ta_i = Ta_i, dim_S_plus_Ts = dim_S_plus_Ts,
                          time_dependent = time_dependent, n_neigh = n_neigh, 
                      w_hidden = w_hidden, Learning_rate = Learning_rate,  n_layer = n_layer, 
                      batch_size = batch_size, max_iteration = max_iteration,  epsilon = epsilon, 
                      spatial = False)
        wi_NS = wi_NS[0]
        wi_NS /= np.mean(wi_NS) 

        DR_V_NS = wi_NS * (Ri + Qi_diff_NS)
        IS_NS = wi_NS * Ri

        """ 3. DR_NO_MF 
        """
        tuples_i = getRegionData(data[i], i, data_neigh, pi, Ts, Ta, mean_field = False)
        Ta_i = arr([pi[j](s = None, random_choose = True) for j in neigh[i]])
        
        n_neigh = len(data_neigh)
        dim_NMF = int(dim_S_plus_Ts / 2 * (n_neigh + 1)) #???
        Qi_diff_NMF, Vi_NMF = computeQV(tuples_i = tuples_i, R = Ri, 
                                      CV_QV = False, penalty_range = penalty_NMF, 
                                      spatial = True, mean_field = False)
        QV_NMF = Vi_NMF[0]
        wi_NMF = getWeight(tuples_i, i, policy0 = behav[i], policy1 = pi[i], dim_S_plus_Ts = dim_NMF,
                   time_dependent = time_dependent, n_neigh = n_neigh, Ta_i = Ta_i,
                  w_hidden = w_hidden, Learning_rate = Learning_rate,  n_layer = n_layer, 
                  batch_size = batch_size, max_iteration = max_iteration,  
                  epsilon = epsilon, spatial = True, mean_field = False)[0]
        DR_V_NMF = wi_NMF * (Ri + Qi_diff_NMF)
        IS_NMF = wi_NMF * Ri
            
        
        """ ending
        """
        values_i = [np.mean(DR_V), QV_V, np.mean(IS_V), 
                    np.mean(DR_V_NS), #QV_NS, np.mean(IS_NS), 
                    np.mean(DR_V_NMF), #QV_NMF, np.mean(IS_NMF), 
#                     np.mean(DR2_V), 
                    V_behav] 
        
        return values_i
    
    if inner_parallel:
        r = arr(parmap(getOneRegionValue, range(N)))
    else:
        r = arr([getOneRegionValue(i) for i in range(N)])
    Vs = np.round(np.mean(r, 0), 3)
    return Vs

##########################################################################################################################################################

""" Transform the data into transition tuples and extract spatial dependence statistics. 
Output:
    tuples_i: a list of transition tuples. 
        tuples_i[t] = [S_it, A_it, R_it, Tsit, Tait, S_i(t+1), Tsi(t+1), pi_Sit_1, T_ait_1_pi]
    data_neigh: 
        a (len-N_j) list of [index, data_at_that_index]
        data_at_that_index is a len-T list, where data_at_that_index[t] is [S_{i,t}, A_{i,t}, R_{i,t}]; 
"""
def getRegionData(data_i, i, data_neigh, pi, Ts, Ta, mean_field = True, time_dependent = False):

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
            Tsit = np.concatenate([a[1][t][0] for a in data_neigh]) # a list (len-#neigh) of state at time t
            Tait = arr([a[1][t][1] for a in data_neigh])
            S_it1 = data_i[t + 1][0]
            A_it1 = data_i[t + 1][1]
            S1_neigh = [a[1][t + 1][0] for a in data_neigh]
            Tsit1 = np.concatenate(S1_neigh)
            pi_Sit_1 = pi[i](S_it1, random_choose = True)
            T_ait_1_pi = arr([pi[a[0]](a[1][t + 1][0], random_choose = True) for a in data_neigh])
        if time_dependent:
            tuple_t[0] = np.append(tuple_t[0], t % 48) # not general enough. specific to our applications.            
            S_it1 = np.append(S_it1, t % 48)
        tuple_t += [Tsit, Tait, S_it1, Tsit1, pi_Sit_1, T_ait_1_pi, A_it1]
        tuples_i.append(tuple_t)
    return tuples_i

##### IS #####################################################################################################################################################

""" Compute the transition tuple density ratios for region i. [Breaking, Lihong]
Input:  
    tuples_i[t] = [S_it, A_it, R_it, Tsit, Tait,  # 0 - 4
                    S_i(t+1), Tsi(t+1), pi_Sit_1, T_ait_1_pi] # 5 - 8
        - what we want: SASR_i = [a list of [S,A,S',R]]; R is useless 
    policy0 = behav[i]
    policy1 = pi[i]
Output: a vector of density ratios.

getWeight(tuples_i, i, policy0 = behav[i], policy1 = pi[i], dim_S_plus_Ts = dim_NMF,
                       time_dependent = time_dependent, n_neigh = n_neigh, Ta_i = Ta_i,
                      w_hidden = w_hidden, Learning_rate = Learning_rate,  n_layer = n_layer, 
                      batch_size = batch_size, max_iteration = max_iteration, 
                      epsilon = epsilon, spatial = True, mean_field = False)
"""
def getWeight(tuples_i, i, policy0, policy1, Ta_i, n_neigh = 8, dim_S_plus_Ts = 3 + 3, time_dependent = False, 
              w_hidden = 10, Learning_rate = 1e-4,  n_layer = 2, 
              batch_size = 64, max_iteration = 1001,  epsilon = 1e-3,
              spatial = True, mean_field = True):
    # prepare transition pairs
    reg_weight = 0 # no penalty in the two-layer NN
    
    # S, A, S', R
    if spatial:
        def concateOne(tuplet):
            # only for our cases
            return [np.concatenate((tuplet[0], tuplet[3]), axis=None), # S
                   [tuplet[1], tuplet[4]], # [A, Ta]
                    np.concatenate((tuplet[5], tuplet[6]), axis=None), # S'
                   tuplet[2]] # R

    else:
        def concateOne(tuplet):
            return [tuplet[0], tuplet[1],
                    tuplet[5], tuplet[2]]
    SASR_i = [concateOne(tuplet) for tuplet in tuples_i]
    SASR_i = [SASR_i] # although we only need 1 layer of list
        
    # Dim and Initialization
    if spatial:
        if mean_field:
            computeWeight = Density_Ratio_kernel(obs_dim = dim_S_plus_Ts, n_layer = n_layer, 
                                     w_hidden = w_hidden, Learning_rate = Learning_rate, reg_weight = reg_weight)
        else:
            computeWeight = Density_Ratio_kernel(obs_dim = 3 * (n_neigh + 1), n_layer = n_layer, 
                                     w_hidden = w_hidden, Learning_rate = Learning_rate, reg_weight = reg_weight)
    else:
        if time_dependent: # general?
            obs_dim = int(dim_S_plus_Ts / 2) + 1
        else:
            obs_dim = int(dim_S_plus_Ts / 2)
        computeWeight = Density_Ratio_kernel(obs_dim = obs_dim, n_layer = n_layer, 
                                     w_hidden = w_hidden, Learning_rate = Learning_rate, reg_weight = reg_weight)

    print_flag = False
    weights = computeWeight.train(SASR_i, policy0, policy1, Ta_i = Ta_i, print_flag = print_flag, 
              batch_size = batch_size, max_iteration = max_iteration, n_neigh = n_neigh, 
               epsilon = epsilon, spatial = spatial, mean_field = mean_field)
    computeWeight.close_Session()
    
    return weights, SASR_i

#### QV ######################################################################################################################################################

""" Value functions [Susan Murphy] with CV
Input:
    tuples_i: required data for the region; 
        tuples_i[t] = [S_it, A_it, R_it, Ts_it, Ta_it, # 0 - 4
                        S_i(t+1), Tsi(t+1), pi_Sit_1, T_ait_1_pi] # 5 - 8
    pi: the target policy

Output:
    Vi: \hat{V}_{i,pi}
    Qi_diff: a vector (len-T) of Q^pi(pi) - Q^pi(behav)
"""
def computeQV(tuples_i, R, n_neigh = None, 
              spatial = True, mean_field = True, 
              CV_QV = False,
              penalty_range = [[0.01], [.01]], K_CV = 3):

    if CV_QV is False:
        penalty = [a[0] for a in penalty_range]
        return computeQV_basic(tuples_i = tuples_i, R = R, penalty = penalty, 
                        spatial = spatial, mean_field = mean_field)
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
                    Bellman_error += computeQV_basic(tuples_i = train_tuples, R = arr([R[i] for i in train_index]), 
                                                     penalty = [mu, lam], 
                                                     n_neigh = n_neigh, spatial = spatial, mean_field = mean_field, 
                                                     validation_set = valid_tuples)
                if Bellman_error < min_Bellman_error:
                    min_Bellman_error = Bellman_error
                    optimal_penalty = [mu, lam]
#         print(spatial, mean_field, optimal_penalty, min_Bellman_error)
        return computeQV_basic(tuples_i = tuples_i, R = R, penalty = optimal_penalty,
                               spatial = spatial, mean_field = mean_field)
                    

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

def computeQV_basic(tuples_i, R, penalty, spatial = True, mean_field = True, 
                    n_neigh = None, 
                    validation_set = None):
    
    ## prepare data
    R = arr([a[2] for a in tuples_i]) # began to use Rit
    T = len(tuples_i)
    mu, lam = penalty
        
    ## get (S,A) pair
    """ RKHS
    """
    # spatial: mean-field or all neigh info
    if spatial:
        if mean_field:
            Z = np.array([np.concatenate((a[0], a[3], [a[1]], [a[4]])) for a in tuples_i]) # T * p. [S, Ts, A, Ta]
            Zstar = np.array([np.concatenate((a[5], a[6], [a[7]], [a[8]])) for a in tuples_i])
            ## kernel distance
            def SA_GRBF(Z, gamma, Z2 = None):
                T, l = Z.shape
                if Z2 is None:
                    Z2 = Z
                    nonsingular = identity(T) * 1e-8
                else:
                    nonsingular = 0
                dim = int(Z.shape[1] // 2 - 1)
                I_A = (Z[:, dim * 2].reshape(-1,1) == Z2[:, dim * 2].reshape(1,-1))
                I_Ta = (Z[:, dim * 2 + 1].reshape(-1,1) == Z2[:, dim * 2 + 1].reshape(1,-1))
                I_A = np.multiply(I_A, I_Ta)
                K = GRBF(Z[:,:(l - 2)], Z2[:,:(l - 2)], gamma) + nonsingular
                return np.multiply(K, I_A)    
        else:
            Z = np.array([np.concatenate((a[0], a[3], [a[1]], a[4])) for a in tuples_i]) # T * p. [S, Ts, A, Ta]
            Zstar = np.array([np.concatenate((a[5], a[6], [a[7]], a[8])) for a in tuples_i])
            ## kernel distance
            def SA_GRBF(Z, gamma, Z2 = None):
                T, l = Z.shape
                if Z2 is None:
                    Z2 = Z
                    nonsingular = identity(T) * 1e-8
                    T2 = T
                else:
                    nonsingular = 0
                    T2 = Z2.shape[0]
                n_neigh = int((l - 4) / 4)
                I_A = (Z[:, 3 * (n_neigh + 1)].reshape(-1,1) == Z2[:, 3 * (n_neigh + 1)].reshape(1,-1))
                I_Ta = randn(T, T2)
                for i in range(T):
                    for j in range(T2):
                        if np.array_equal(Z[i, (3 * (n_neigh + 1) + 1):], Z2[j, (3 * (n_neigh + 1) + 1):]): 
                            I_Ta[i, j] = 1
                I_A = np.multiply(I_A, I_Ta)
                K = GRBF(Z[:, :(3 * (n_neigh + 1))], Z2[:, :(3 * (n_neigh + 1))], gamma) + nonsingular
                return np.multiply(K, I_A)    
    # single agent
    else:
        Z = np.array([np.concatenate((a[0], [a[1]])) for a in tuples_i]) # T * p. [S, A, Ts, Ta]
        Zstar = np.array([np.concatenate((a[5], [a[7]])) for a in tuples_i])
        ## kernel distance
        def SA_GRBF(Z, gamma, Z2 = None):
            T, l = Z.shape
            if Z2 is None:
                Z2 = Z
                nonsingular = identity(T) * 1e-8
            else:
                nonsingular = 0
            dim = int(Z.shape[1] - 1)
            I_A = (Z[:, dim].reshape(-1,1) == Z2[:, dim].reshape(1,-1))
            K = GRBF(Z[:,:(l - 1)], Z2[:,:(l - 1)], gamma)  + nonsingular
            return np.multiply(K, I_A)    
    
    """ gammas for RKHS
    """
    Z_tilde = np.vstack((Z, Zstar))
    if spatial:
        if mean_field:
            gamma_g = 1 / (2 * np.median(pdist(Z[:,:(Z.shape[1]-2)]))**2)
            gamma_q = 1 / (2 * np.median(pdist(Z_tilde[:,:(Z_tilde.shape[1]-2)]))**2)
        else:
            T, l = Z.shape
            n_neigh = int((l - 4) / 4)
            
            gamma_g = 1 / (2 * np.median(pdist(Z[:, :(3 * (n_neigh + 1))]))**2)
            gamma_q = 1 / (2 * np.median(pdist(Z_tilde[:, :(3 * (n_neigh + 1))]))**2)
    else:
        gamma_g = 1 / (2 * np.median(pdist(Z[:,:(Z.shape[1]-1)]))**2)
        gamma_q = 1 / (2 * np.median(pdist(Z_tilde[:,:(Z_tilde.shape[1]-1)]))**2)
        
    """ main
    """
    Kg = SA_GRBF(Z, gamma_g)
    KQ = SA_GRBF(Z_tilde, gamma_q)

    ## Idnetity vec/mat
    C = np.hstack((-identity(T),identity(T)))       
    vec1, I = ones(T).reshape(-1,1), identity(T)
    
    E_right_bef_inverse = Kg + T * mu * I # RHS of E
    
    CKQ_1 = np.hstack((C.dot(KQ), -vec1))
    ECKQ1 = Kg.T.dot(solve(E_right_bef_inverse, CKQ_1)) # E[CK_Q,-1]
    
    left = (ECKQ1.T.dot(ECKQ1) + np.vstack((np.hstack((T * lam * KQ, zeros((2 * T, 1)))), zeros((1, 2 * T + 1))))) # Left part of (\hat{\alpha}, \hat{\eta})
    right = ECKQ1.T.dot(Kg.dot(solve(E_right_bef_inverse, R))) # Right part of (\hat{\alpha}, \hat{\eta})
    if validation_set is not None:
        alpha_eta = np.linalg.lstsq(left, np.expand_dims(right,1))[0]
    else:
        alpha_eta = -solve(left, np.expand_dims(right,1)) 
    alpha = alpha_eta[:(len(alpha_eta) - 1)]
    Vi = eta = alpha_eta[-1]
    
    """ validation
    """
    if validation_set is not None: # used for Cross-validation
        A_set = set([a[1] for a in tuples_i])
        Ta_set = np.unique(arr([a[4] for a in tuples_i]), axis=0)
        R = arr([a[2] for a in validation_set]) # dim?
        # not standardization yet.
        if not spatial:
            SA_t = np.array([np.concatenate((a[0], [a[1]])) for a in validation_set])            
        else:
            if mean_field: 
                SA_t = np.array([np.concatenate((a[0], a[3], [a[1]], [a[4]])) for a in validation_set]) # [S, Ts, A, Ta]
            else:
                SA_t = np.array([np.concatenate((a[0], a[3], [a[1]], a[4])) for a in validation_set]) # [S, Ts, A, Ta]
        T = SA_t.shape[0]

#         QSA = alpha.T.dot(GRBF(Z_tilde, SA_t, gamma_q)).T
        QSA = alpha.T.dot(SA_GRBF(Z = Z_tilde, gamma = gamma_q, Z2 = SA_t)).T
        QSA1 = 0 * QSA
        # not CV for non-MF yet
        if not spatial:
            for action in A_set:
                SA_t1 = np.array([np.concatenate((a[5], [action])) for a in validation_set])  
                QSA1_a = alpha.T.dot(SA_GRBF(Z = Z_tilde, gamma = gamma_q, Z2 = SA_t1)).T
                QSA1 += QSA1_a
        else:
            SA_t1 = []
            count = 0
            for action in A_set:
                for Ta in Ta_set:
                    if mean_field:
                        SA_t1.append(arr([np.concatenate([a[5], a[6], [action], [Ta] ]) for a in validation_set])) # action
                    else:
                        SA_t1.append(arr([np.concatenate([a[5], a[6], [action], Ta]) for a in validation_set])) # action
                    count += 1
            SA_t1 = np.vstack(SA_t1)
            QSA1 = alpha.T.dot(SA_GRBF(Z = Z_tilde, gamma = gamma_q, Z2 = SA_t1)).T
            QSA1 = np.sum(QSA1.reshape((count, -1)), 0)
            
        bellman_errors = squeeze(R) + squeeze(QSA1) - eta - squeeze(QSA)
        
        kernel = DotProduct() + WhiteKernel()
        gpr = GaussianProcessRegressor(kernel=kernel,
            random_state=0, normalize_y = True).fit(SA_t, bellman_errors)
        return np.mean(gpr.predict(SA_t)**2)
#     """ not validation
#     """   
    else: # calculate the final value
        Qvalues = alpha.T.dot(KQ)
        Qi_diff = Qvalues[0, T:] - Qvalues[0, :T] # Q^* - Q
        return Qi_diff, Vi