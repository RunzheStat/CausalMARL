from _uti_basic import *
from _utility import *
from weight import *
from simu_funs import *
##########################################################################################################################################################
##########################################################################################################################################################


def V_DR(data, pi, behav, adj_mat, Ts, Ta, penalty = [[1e-2, 1e-2]], 
         dim_S_plus_Ts = 3 + 3, n_cores = 10,
            time_dependent = False, den_fun = None, 
            w_hidden = 30, Learning_rate = 1e-3,  n_layer = 2, CV_QV = False, 
            batch_size = 32, max_iteration = 1001, test_num = 0, epsilon = 1e-6, inner_parallel = False,
            competing_methods = True,
        isValidation = False): 
    """ Compute the average reward estimate using DR (and other methods) with one dataset
    Input: 
        data: a len-N list. data[i] is a len-T list, where data[i][t] is [S_{i,t}, A_{i,t}, R_{i,t}]; 
            for now, we only use the state in data[i][T]
        policies:
            pi: a list (len-N) of the target policies
            behav: a list (len-N) of the behaviour policy
            pi(s, a) = a probability # although in didi, may not be the case
        adj_mat: binary adjacent matrix
        Ts, Ta: required spatial dependence functions
        competing_methods: whether calculate the values with the other four methods
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
        n_neigh = len(neigh[i])
        Ri = arr([a[2] for a in data[i]])[:T]
        tuples_i = getRegionData(data[i], i, data_neigh, pi, Ts, Ta, mean_field = True, time_dependent = time_dependent)
        
        
        Ta_i = Ta_disc(np.mean([pi[j](s = None, random_choose = True) for j in neigh[i]]))
        
        """ our method """
        Qi_diff, Vi = computeQV(tuples_i = tuples_i, R = R, 
                                CV_QV = CV_QV, penalty_range = penalty)
        r = getWeight(tuples_i, i, policy0 = behav[i], policy1 = pi[i], Ta_i = Ta_i, dim_S_plus_Ts = dim_S_plus_Ts,
                       time_dependent = time_dependent, n_neigh = n_neigh,
                      w_hidden = w_hidden, Learning_rate = Learning_rate,  n_layer = n_layer, 
                      batch_size = batch_size, max_iteration = max_iteration, test_num = test_num, 
                      epsilon = epsilon, isValidation = isValidation)

        wi = r[0]
        wi /= np.mean(wi)
        DR_V = wi * (Ri + Qi_diff)
        
        """ DEBUG with MC_based weight estimates
        """
        if isValidation:
            sample_states = [SASR[0] for SASR in r[1][0]]
            sample_SA = [np.array([np.concatenate((a[0], [a[1]], a[3], [a[4]])) for a in tuples_i]), 
                         np.array([np.concatenate((a[5], [a[7]], a[6], [a[8]])) for a in tuples_i])]
            MC_den_ratio, MC_Qi_diff = den_fun(sample_states, sample_SA, i)
            MC_den_ratio /= np.mean(MC_den_ratio)
            print("WEIGHT estimation error (std of MC/est, median of diff, R2): \n", 
                  np.round([np.std(MC_den_ratio),  np.std(wi), 
                            np.median(np.abs(MC_den_ratio - wi)), 
                            R2(MC_den_ratio, wi)], 2), "\n")
            print("Q_diff estimation error (std of MC/est, median of diff, R2): \n", 
                  np.round([np.std(MC_Qi_diff), np.std(Qi_diff),  
                            np.median(np.abs(MC_Qi_diff - Qi_diff)), 
                            R2(MC_Qi_diff, Qi_diff)], 2), "\n")
        
        """ COMPETING METHODS
        move these methods out; but they are very simple
        """
        if competing_methods:            
            """ 1. with MARL """
            IS_V = wi * Ri
            QV_V = Vi[0] 
            DR2_V = IS_V + QV_V - DR_V # the DR proposed by that paper
            V_behav = np.mean(Ri) # the average reward of behavious policy (no difference?)
            """ WAIT
            """
        
            """ 2. DR + w/o spatial """
            Qi_diff_NS, Vi_NS = computeQV(tuples_i = tuples_i, R = R, 
                                CV_QV = CV_QV, penalty_range = penalty, spatial = False)
            QV_NS = Vi_NS[0]
            
            wi_NS = getWeight(tuples_i, i, policy0 = behav[i], policy1 = pi[i], Ta_i = Ta_i, dim_S_plus_Ts = dim_S_plus_Ts,
                              time_dependent = time_dependent, n_neigh = n_neigh, 
                          w_hidden = w_hidden, Learning_rate = Learning_rate,  n_layer = n_layer, 
                          batch_size = batch_size, max_iteration = max_iteration, test_num = test_num, epsilon = epsilon, 
                          spatial = False)
            wi_NS = wi_NS[0]
            wi_NS /= np.mean(wi_NS) 

            DR_V_NS = wi_NS * (Ri + Qi_diff_NS)
            IS_NS = wi_NS * Ri
            
            values_i = [np.mean(DR_V), QV_V, np.mean(IS_V), np.mean(DR_V_NS), QV_NS, np.mean(IS_NS), V_behav] # np.mean(DR2_V), 
                        
            return values_i
        else:
            return np.mean(DR_V)
    
    if inner_parallel:
        r = arr(parmap(getOneRegionValue, range(N), n_cores))
    else:
        r = arr([getOneRegionValue(i) for i in range(N)])
            
    Vs = np.round(np.mean(r, 0), 3)
    return Vs

##########################################################################################################################################################

""" Transform the data into transition tuples and extract spatial dependence statistics. 
Output:
    tuples_i: a list of transition tuples. 
        tuples_i[t] = [S_it, A_it, R_it, Tsit, Tait, S_i(t+1), Tsi(t+1), pi_Sit_1, T_ait_1_pi]
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

##### IS #####################################################################################################################################################

""" Compute the transition tuple density ratios for region i. [Breaking, Lihong]
Input:  
    tuples_i[t] = [S_it, A_it, R_it, Tsit, Tait,  # 0 - 4
                    S_i(t+1), Tsi(t+1), pi_Sit_1, T_ait_1_pi] # 5 - 8
        - what we want: SASR_i = [a list of [S,A,S',R]]; R is useless 
    policy0 = behav[i]
    policy1 = pi[i]
Output: a vector of density ratios.
"""
def getWeight(tuples_i, i, policy0, policy1, Ta_i, dim_S_plus_Ts, time_dependent = False, 
              w_hidden = 10, Learning_rate = 1e-4,  n_layer = 2, 
              n_neigh = 8, 
              batch_size = 64, max_iteration = 1001, test_num = 0, epsilon = 1e-3,
              spatial = True, isValidation = False):
    # prepare transition pairs
    reg_weight = 0 # no penalty in the two-layer NN
    
    # S, A, S', R
    if spatial:
        def concateOne(tuplet):
            # only for our cases
            return [np.concatenate((tuplet[0], tuplet[3]), axis=None), 
                   [tuplet[1], tuplet[4]], # [A, Ta]
                    np.concatenate((tuplet[5], tuplet[6]), axis=None), 
                   tuplet[2]]
#             return [np.concatenate((tuplet[0], tuplet[3]), axis=None), 
#                    tuplet[4],
#                     np.concatenate((tuplet[5], tuplet[6]), axis=None), 
#                    tuplet[8]]
    else:
        def concateOne(tuplet):
            return [tuplet[0], tuplet[1],
                    tuplet[5], tuplet[2]]
    SASR_i = [concateOne(tuplet) for tuplet in tuples_i]
    SASR_i = [SASR_i] # although we only need 1 layer of list
        
    # Dim and Initialization
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
    """
    data is from the behaviour
    def behav(s, a):
        return 0.5
    def pi_i(s, a = 0, random_choose = False, reward = reward):
            if random_choose:
                return reward
            else:
                return int(a == reward)
    """
#     if i in [0, 6, 12]:
#         print_flag = True
#     else:
    print_flag = False
    weights = computeWeight.train(SASR_i, policy0, policy1, Ta_i = Ta_i, print_flag = print_flag, 
              batch_size = batch_size, max_iteration = max_iteration, n_neigh = n_neigh, 
              test_num = test_num, epsilon = epsilon, only_state = isValidation, spatial = spatial)
    computeWeight.close_Session()
    
    return weights, SASR_i

#### QV ######################################################################################################################################################

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
def computeQV(tuples_i, R, spatial = True, CV_QV = False,
              penalty_range = [[0.01], [.01]], K_CV = 3):

    if CV_QV is False:
        penalty = [a[0] for a in penalty_range]
        return computeQV_basic(tuples_i = tuples_i, R = R, penalty = penalty, 
                        spatial = spatial)
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
                                    spatial = spatial, 
                                    validation_set = valid_tuples)
#                     validation_set = [valid_tuples, arr([R[i] for i in valid_tuples])]
                if Bellman_error < min_Bellman_error:
                    min_Bellman_error = Bellman_error
                    optimal_penalty = [mu, lam]
        return computeQV_basic(tuples_i = tuples_i, R = R, penalty = optimal_penalty, spatial = spatial)
                    

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
def computeQV_basic(tuples_i, R, penalty, spatial = True, 
                    validation_set = None):
    
    ## prepare data
    R = arr([a[2] for a in tuples_i]) # began to use Rit
    T = len(tuples_i)
    mu, lam = penalty
    A_set = set([a[1] for a in tuples_i])
    
    ## get (S,A) pair
    if spatial:
        Z = np.array([np.concatenate((a[0], a[3], [a[1]], [a[4]])) for a in tuples_i]) # T * p. [S, Ts, A, Ta]
        Zstar = np.array([np.concatenate((a[5], a[6], [a[7]], [a[8]])) for a in tuples_i])
        ## kernel distance
        def SA_GRBF(Z, gamma):
            T, l = Z.shape
            dim = int(Z.shape[1] // 2 - 1)
            I_A = (Z[:, dim * 2].reshape(-1,1) == Z[:, dim * 2].reshape(1,-1))
            I_Ta = (Z[:, dim * 2 + 1].reshape(-1,1) == Z[:, dim * 2 + 1].reshape(1,-1))
            I_A = np.multiply(I_A, I_Ta)
            K = GRBF(Z[:,:(l - 2)], Z[:,:(l - 2)], gamma) + identity(T) * 1e-8
            return np.multiply(K, I_A)        
    else:
        Z = np.array([np.concatenate((a[0], [a[1]])) for a in tuples_i]) # T * p. [S, A, Ts, Ta]
        Zstar = np.array([np.concatenate((a[5], [a[7]])) for a in tuples_i])
        ## kernel distance
        def SA_GRBF(Z, gamma):
            T, l = Z.shape
            dim = int(Z.shape[1] - 1)
            I_A = (Z[:, dim].reshape(-1,1) == Z[:, dim].reshape(1,-1))
            K = GRBF(Z[:,:(l - 1)], Z[:,:(l - 1)], gamma) + identity(T) * 1e-8
            return np.multiply(K, I_A)    
    
    Z_tilde = np.vstack((Z, Zstar))
    if spatial:
        gamma_g = 1 / (2 * np.median(pdist(Z[:,:(Z.shape[1]-2)]))**2)
        gamma_q = 1 / (2 * np.median(pdist(Z_tilde[:,:(Z_tilde.shape[1]-2)]))**2)
    else:
        gamma_g = 1 / (2 * np.median(pdist(Z[:,:(Z.shape[1]-1)]))**2)
        gamma_q = 1 / (2 * np.median(pdist(Z_tilde[:,:(Z_tilde.shape[1]-1)]))**2)
        
    Kg = SA_GRBF(Z, gamma_g)
    KQ = SA_GRBF(Z_tilde, gamma_q)
    # centeralization, p11
#     ZTstar = np.mean(Z_tilde, 0)
#     KQ = KQ - GRBF(Z_tilde, ZTstar.reshape(1, -1), gamma_q) - GRBF(ZTstar.reshape(1, -1), 
#                                                                    Z_tilde, gamma_q) - GRBF(ZTstar.reshape(1, -1), ZTstar.reshape(1, -1), gamma_q)[0][0]
    
    ## Idnetity vec/mat
    C = np.hstack((-identity(T),identity(T)))       
    vec1, I = ones(T).reshape(-1,1), identity(T)
    
    E_right_bef_inverse = Kg + T * mu * I # RHS of E
    
    CKQ_1 = np.hstack((C.dot(KQ), -vec1))
    ECKQ1 = Kg.T.dot(solve(E_right_bef_inverse, CKQ_1)) # E[CK_Q,-1]
    
    left = (ECKQ1.T.dot(ECKQ1) + np.vstack((np.hstack((T * lam * KQ, zeros((2 * T, 1)))), zeros((1, 2 * T + 1))))) # Left part of (\hat{\alpha}, \hat{\eta})
    right = ECKQ1.T.dot(Kg.dot(solve(E_right_bef_inverse, R))) # Right part of (\hat{\alpha}, \hat{\eta})
    alpha_eta = -solve(left, np.expand_dims(right,1)) 
    alpha = alpha_eta[:(len(alpha_eta) - 1)]
    Vi = eta = alpha_eta[-1]
    
    """ no centeralization for KQ [? for below?]
    """
    if validation_set is not None: # used for Cross-validation
        R = arr([a[2] for a in validation_set]) # dim?
        # not standardization yet.
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
        bellman_errors = R + QSA1 - eta - QSA
        kernel = DotProduct() + WhiteKernel()
        gpr = GaussianProcessRegressor(kernel=kernel,
            random_state=0).fit(SA_t, bellman_errors)
        return np.mean(gpr.predict(SA_t)**2)  
        
    else: # calculate the final value
        Qvalues = alpha.T.dot(KQ)
        Qi_diff = Qvalues[0, T:] - Qvalues[0, :T] # Q^* - Q
        return Qi_diff, Vi

##########################################################################################################################################################
##########################################################################################################################################################
    #- GRBF(Z, Zstar.reshape(1, -1), gamma_g) - GRBF(Zstar.reshape(1, -1), Z, gamma_g)  - GRBF(Zstar.reshape(1, -1), Zstar.reshape(1, -1), gamma_g)[0][0]

""" 3. DR w/o mean field (not yet) """
#             tuples_i = getRegionData(data[i], i, data_neigh, pi, Ts, Ta, mean_field = False)
#             n_neigh = len(data_neigh)
#             dim_NMF = int(dim_S_plus_Ts / 2 * (n_neigh + 1))
#             Qi_diff_NS, Vi_NS = computeQV(tuples_i, R, penalty, dim_S_plus_Ts = dim_NMF, spatial = True)
#             wi_NS = getWeight(tuples_i, policy0 = behav[i], policy1 = pi[i], dim_S_plus_Ts = dim_NMF,
#                           w_hidden = w_hidden, Learning_rate = Learning_rate,  n_layer = n_layer, 
#                           batch_size = batch_size, max_iteration = max_iteration, test_num = test_num, epsilon = epsilon, 
#                           spatial = True)
#             DR_V_NMF = wi_NS * (Ri + Qi_diff_NS - np.squeeze(arr(list(itertools.repeat(Vi_NS, T))))) + Vi
#             return [np.mean(DR_V), np.mean(IS_V), Susan_V, np.mean(DR_V_NS), np.mean(DR_V_NMF)]


    
    # normalization: NA
#     for i in range(4):
#         sd = np.std(arr([a[i] for a in SASR_i]), 0)
#         print("sd:", sd)
#         for j in range(len(SASR_i)):
#             SASR_i[j][i] /= sd
#             if j == 0:
#                 print(SASR_i[j][i])
