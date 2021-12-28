#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main functions for the test proposed in the paper "Does MDP Fit the Data?". Refer to the Algorithm 1 and 2 therein.
"""

##########################################################################
#%% 
from _QRF import *
from _uti_basic import *
from _utility import *

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

##########################################################################
# %%
n_jobs = multiprocessing.cpu_count()
param_grid = {'max_depth': [2, 4, 6, 8], 'min_samples_leaf': [5, 10, 20]}

##########################################################################
#%% Algorithm 1
class MarkovTester():
    def __init__(self, data, J = 1,
        B = 200, Q = 10, L = 3, 
        paras="CV", n_trees = 200, 
        print_time = False,
        include_reward = False, fixed_state_comp = None, 
        method = "QRF"):
        """
        The main test function

        Parameters
        ----------
        data: the observed trajectories. A len-N list of [X, A], where X and A are T * dim arrays.
        J: the null hyphothesis that the MDP is lag-J. Donoted as k in the paper
        B, Q: required  hyperparameters; The definition of Q is slightly different with the paper. Q_here = Q_paper + 2
        paras: the parameters [max_depth, min_samples_leaf] used in the random forests.
        n_trees: the number of trees used in the random forests
        print_time: whether or not to print out the time cost for each part
        include_reward: whether or not to include the R_t as part of X_t for our testing
        fixed_state_comp: to resolve the duplicate S problem in the TIGER
        method: the estimators used for the conditional characteristic function estimation.

        Returns
        -------
        p-values

        * Old data: the observed trajectories. A len-N list of [X, A], where X and A are T * dim arrays.
        * Our data: a len-N list of $[(MF^S_{t}, MF^A_{t}, S_{t}, A_t), (R_{t}, S_{t+1}), (S_{t,full}, A_{t,full})]$, where each element is an $T \times dim$ matrix
        """
        N = len(data)
        def normalize(a):
            aa = np.std(a, axis = 0)
            if sum(aa == 0):
                return a
            else:
                return a / aa
        data = [[normalize(a) for a in b] for b in data]
        
        T = data[0][0].shape[0]
        a = now()
        lam = lam_est(data = data, J = J, B = B, Q = Q, paras = paras, n_trees = n_trees, 
                      include_reward = include_reward, L = L, 
                      fixed_state_comp = fixed_state_comp, method = method)
        r, pValues = [], []
        Sigma_q_s = Sigma_q(lam)  # a list (len = Q-1) 2B * 2B.
        if print_time:
            print("RF:", now() - a)
        a = now()
        S = S_hat(lam = lam, dims = [N, T], J = J)  # Construct the test statistics
        pValues = bootstrap_p_value(Sigma_q_s, rep_times = int(1e3), test_stat = S)  # Construct the resampling-based c.v.
        if print_time:
            print("Bootstrap:", now() - a)
        self.S = S
        self.Sigma_q_s = Sigma_q_s
        self.pValues = pValues


        
##########################################################################
#%% Getting data. Helper functions
def get_pairs( data, is_forward, J = 1, as_array = 1, include_reward = 0, fixed_state_comp = None):
    """
    get [(x_{t-1},a_{t-1}),x_t] or [(x_t,a_t),(x_{t-1},a_{t-1})] pairs, only for training[can not distinguish patients]

    forward: indicator
    as_array: by default, into pred/response array
    

    * Old data: the observed trajectories. A len-N list of [X, A], where X and A are T * dim arrays.
    * Our data: a len-N list of $[(MF^S_{t}, MF^A_{t}, S_{t}, A_t), (R_{t}, S_{t+1}), (S_{t,full}, A_{t,full})]$, where each element is an $T \times dim$ matrix
    
    """
    def get_pairs_one_traj(i, is_forward, J):
        """
        do one patient for <get_pairs>, get trainiinig data
        patient = [X,A]
        X = T * d_x, A = T * d_a
        """
        patient = data[i]
#         if include_reward:
#             X, A, R = patient
#         else:
#             X, A = patient
        SA_now, SR_next, SA_full = patient
        T = SA_now.shape[0]
        r = []
        dx = SA_now.shape[1]

        for t in range(T - J):
            if is_forward:
                pair = [SA_now[t, :], SR_next[t + J, :]]
            else:
                pair = [SA_now[t, :], SA_full[t, :]]
                
            r.append(pair)
        return r

    # get pairs for each patient and put together
    r = flatten([get_pairs_one_traj(i, is_forward, J)
                 for i in range(len(data))])
    if as_array:
        r = [np.vstack([a[0] for a in r]), np.vstack([a[1] for a in r])]
    return r


def get_test_data(test_data, J=1, fixed_state_comp=None):
    """
    Get testing predictors
    """
    def patient_2_predictors(i, J=1):
        """
        XA: T * (d_x + d_a)
        Return: T * ((d_x + d_a) * J)
        """
        patient = test_data[i]
        XA = patient[0] #np.hstack([patient[0], patient[1]])
#         T = XA.shape[0]
        r = XA.copy()
#         for j in range(1, J): 
#             r = np.hstack([r, roll(XA, -j, 0)])
        return r

    return np.vstack([patient_2_predictors(i, J)
                      for i in range(len(test_data))])

##########################################################################
# Functions for estimating the CCF and constructing the conditional covariances. Step 2 - 3 of Algorithm 1.

# %% Conditional covariance lam construction
def lam_est(data, J, B, Q, L = 3, 
            paras = [3, 20], n_trees = 200, include_reward = 0, fixed_state_comp = None, method = "QRF"):
    """
    construct the pointwise cov lam (for both test stat and c.v.), by combine the two parts (estimated and observed)

    Returns
    -------
    lam: (Q-1)-len list of four lam matrices (n * T-q * B)
    
    * Old data: the observed trajectories. A len-N list of [X, A], where X and A are T * dim arrays.
    * Our data: a len-N list of $[(MF^S_{t}, MF^A_{t}, S_{t}, A_t), (R_{t}, S_{t+1}), (S_{t,full}, A_{t,full})]$, where each element is an $T \times dim$ matrix

    """
    dims = [data[0][i].shape[1] for i in range(3)]
#     dx, da = data[0][0].shape[1], data[0][1].shape[1]
#     if fixed_state_comp is not None:
#         dx += 1

    # generate uv
    rseed(0); npseed(0)

    uv = [randn(B, dims[1]), randn(B, dims[2])] # future (R_{t}, S_{t+1}), past (S_{t,full}, A_{t,full})
    
    # estimate characteristic values (cross-fitting): phi_R, psi_R, phi_I,
    # psi_I
    estimated = cond_char_vaule_est(data = data, uv = uv,
                                    paras = paras, n_trees = n_trees, L = L, J = J, 
                                    include_reward = include_reward, fixed_state_comp = fixed_state_comp, 
                                   method = method)  # ,obs_ys
    if paras == "CV_once":
        CV_paras = estimated
        return CV_paras
    else:
        estimated_cond_char = estimated
        # cos and sin in batch. (n*T*dx) * (dx* B)  = n * T * B:
        # c_X,s_X,c_XA,s_XA
        observed_cond_char = obs_char(data = data, uv = uv, 
            include_reward = include_reward, fixed_state_comp = fixed_state_comp)
        # combine the above two parts to get cond. corr. estimation.
        lam = lam_formula(estimated_cond_char, observed_cond_char, J, Q)
        return lam


def cond_char_vaule_est(data, uv,
        paras = "CV_once", n_trees = 200, L = 3, 
        J = 1, include_reward = 0, fixed_state_comp = None, method = "QRF"):
    """
    Cross-fitting-type prediction of the cond. char "values"

    Returns
    -------
    phi_R, phi_I, psi_R, psi_I values as [n * T * B] tensors.
    """
    T = data[0][0].shape[0]
    n = N = len(data)
    B = uv[0].shape[0]
    char_values, obs_ys = [np.zeros([n, T, B]) for i in range(4)], [
        np.zeros([n, T, B]) for i in range(4)]
    K = L  # num of cross-fitting
    kf = KFold(n_splits=K)
    kf.get_n_splits(zeros(n))

    # Just to get CV-based paras
    if paras == "CV_once":
        for train_index, test_index in kf.split(
                data):  # only do this one time to get paras by using CV
            if fixed_state_comp:
                true_state_train = [fixed_state_comp[i] for i in train_index]
            else:
                true_state_train = None
            train_data, test_data = [data[i] for i in train_index], [data[i] for i in test_index]
            CV_paras = char_fun_est(train_data = train_data,
                paras = "CV_once", n_trees = n_trees, uv = uv, J = J,
                include_reward=include_reward, fixed_state_comp=true_state_train)
            return CV_paras

    # estimate char values by cross-fitting
    for train_index, test_index in kf.split(data):
        if fixed_state_comp:
            true_state_train = [fixed_state_comp[i] for i in train_index]
            true_state_test = [fixed_state_comp[i] for i in test_index]
        else:
            true_state_train, true_state_test = None, None
        train_data, test_data = [data[i] for i in train_index], [data[i] for i in test_index]
        test_pred = get_test_data(test_data = test_data, J = J, fixed_state_comp = true_state_test)
        a = now()
        

        char_funs = char_fun_est(train_data=train_data, paras=paras, n_trees = n_trees, 
                                 uv=uv, J=J, include_reward=include_reward,
                                 fixed_state_comp=true_state_train) # a list of four estimated fun

        for i in range(2):  # forward / backward
            r = char_funs[i].predict(test_pred, uv[i])  # return: char_est_cos, char_est_sin
            char_values[0 + i][test_index] = r[0].reshape((len(test_index), T, B))
            char_values[2 + i][test_index] = r[1].reshape((len(test_index), T, B))
    return char_values 


def char_fun_est(
        train_data,
        paras=[3, 20], n_trees = 200, uv = 0, J = 1, include_reward = 0, fixed_state_comp = None):
    """
    For each cross-fitting-task, use QRF to do prediction

    paras == "CV_once": use CV_once to fit
    get_CV_paras == True: just to get paras by using CV

    Returns
    -------
    a list of four estimated fun, and a list of four true y vectors
    """

    char_funs = []
    X1, y1 = get_pairs(train_data, is_forward = 1, J = J,
                       include_reward = include_reward, fixed_state_comp = fixed_state_comp)
    X2, y2 = get_pairs(train_data, is_forward = 0, J = J,
                       include_reward = include_reward, fixed_state_comp = fixed_state_comp)

    X, y = [X1, X2], [y1, y2]

    if paras in ["CV", "CV_once"]:
        for i in range(2):
            rfqr = RandomForestQuantileRegressor(random_state=0, n_estimators = n_trees)
            gd = GridSearchCV(estimator = rfqr, param_grid = param_grid, 
                              cv = 5, n_jobs = n_jobs, verbose=0)
            gd.fit(X[i], y[i])
            best_paras = gd.best_params_

            if paras == "CV_once":  # only return forward
                return [best_paras['max_depth'], best_paras['min_samples_leaf']]

            elif paras == "CV":
                #print("best_paras:", best_paras)
                # use the optimal paras and the whole dataset
                rfqr1 = RandomForestQuantileRegressor(
                    random_state=0,
                    n_estimators = n_trees, 
                    max_depth=best_paras['max_depth'],
                    min_samples_leaf=best_paras['min_samples_leaf'], 
                    n_jobs = n_jobs)
                char_funs.append(rfqr1.fit(X[i], y[i]))

    else:  # pre-specified paras
        max_depth, min_samples_leaf = paras
        for i in range(2):
            char_funs.append(
                RandomForestQuantileRegressor(
                    random_state=0, n_estimators = n_trees, 
                    max_depth = max_depth, min_samples_leaf = min_samples_leaf, 
                    n_jobs = n_jobs).fit( X[i], y[i]))

    return char_funs


def obs_char(data, uv, include_reward, fixed_state_comp=None):
    """
    Batchwise calculation for the cos/sin terms, used to define lam
    (n*T*dx) * (dx* B)  = n * T * B
    
        * Old data: the observed trajectories. A len-N list of [X, A], where X and A are T * dim arrays.
        * Our data: a len-N list of $[(MF^S_{t}, MF^A_{t}, S_{t}, A_t), (R_{t}, S_{t+1}), (S_{t,full}, A_{t,full})]$, where each element is an $T \times dim$ matrix

    """
    
    """ TODO """
#     T = data[0][0].shape[0]
#     X_mat = np.array([a[1] for a in data])
#     A_mat = np.array([a[2] for a in data])
#     XA_mat = np.concatenate([X_mat, A_mat], 2)
    
#     if include_reward:
#         R_mat = np.array([a[2] for a in data])
#         XR_mat = np.concatenate([X_mat, R_mat], 2)
#         S = [XR_mat, XA_mat]
#     else:
#         S = [X_mat, XA_mat]
        
    S = [np.array([a[1] for a in data]), np.array([a[2] for a in data])]
    r = []
    for i in range(2):
        temp = S[i].dot(uv[i].T)
        r += [cos(temp), sin(temp)]
    return r


def lam_formula(char_values, c_s_values, J, Q):
    """
    implement the 4 lam formula (point cond. cov)
    # char_values: predict t + J and t - 1; # len-4 list, the  element is len-n [T_i, B]
    Inputs:
        char_values: predicted values, at point t, they are [t, …, t + J - 1] -> [t - 1] and [t + J]
        c_s_values: observed values, t is just t
    Outputs:
        lam: (Q-1)-len list with every entry as [four (n * T-q * B) matries about lam values]
    """
    phi_R, psi_R, phi_I, psi_I = char_values
    c_X, s_X, c_XA, s_XA = c_s_values

    # forward, t is the residual at time t
    
    left_cos_R = c_X - phi_R
    left_sin_I = s_X - phi_I
    # backward, t is the residual at time t
    right_cos_R = c_XA - psi_R
    right_sin_I = s_XA - psi_I

#     left_cos_R = c_X - roll(phi_R, J, 1)
#     left_sin_I = s_X - roll(phi_I, J, 1)
#     # backward, t is the residual at time t
#     right_cos_R = c_XA - roll(psi_R, -1, 1)
#     right_sin_I = s_XA - roll(psi_I, -1, 1)

    lam = []
    for q in range(2, Q + 1):
        shift = q + J - 1
        startT = q + J - 1
        lam_RR = multiply(
            left_cos_R, right_cos_R)[
            :, startT:, :]
        lam_II = multiply(
            left_sin_I, right_sin_I)[
            :, startT:, :]
        lam_IR = multiply(
            left_sin_I, right_cos_R)[
            :, startT:, :]
        lam_RI = multiply(
            left_cos_R, right_sin_I)[
            :, startT:, :]
        lam.append([lam_RR, lam_II, lam_IR, lam_RI])

#     for q in range(2, Q + 1):
#         shift = q + J - 1
#         startT = q + J - 1
#         lam_RR = multiply(
#             left_cos_R, roll(
#                 right_cos_R, shift, 1))[
#             :, startT:, :]
#         lam_II = multiply(
#             left_sin_I, roll(
#                 right_sin_I, shift, 1))[
#             :, startT:, :]
#         lam_IR = multiply(
#             left_sin_I, roll(
#                 right_cos_R, shift, 1))[
#             :, startT:, :]
#         lam_RI = multiply(
#             left_cos_R, roll(
#                 right_sin_I, shift, 1))[
#             :, startT:, :]
#         lam.append([lam_RR, lam_II, lam_IR, lam_RI])
    return lam


##########################################################################
# %% The final test statistics and p-values
# only rely on estimated cond. cov [est cond. char v.s. obs cond. char]
# has nothing to do with the char estimation part
##########################################################################

#%% part 2 of Step 3 for Algorithm 1
def S_hat(lam, dims, J = 1):
    """
    Construct the test stat S based on cond. covs.
        1. construct (Q-1 * B) Gammas from lam(sample lag-q covariance functions)
        2. Step3 - aggregate to get S_hat

    Inputs:
        lam: (Q-1)-len list of four lam matrices (n * T-q * B)

    Ourputs:
    """
    Gamma = [np.array([np.mean(a[i], (0, 1)) for a in lam]) for i in range(4)]
    Gamma_R = Gamma[0] - Gamma[1]  # Gamma_RR - Gamma_II
    Gamma_I = Gamma[2] + Gamma[3]  # Gamma_IR + Gamma_RI

    N, T = dims
    Q = Gamma_R.shape[0] + 1
    B = Gamma_R.shape[1]
    r = []

    for q in range(2, Q + 1):
        c = sqrt(N * (T + 1 - q - J))
        r.append(c * max(max(Gamma_R[q - 2, :]), max(Gamma_I[q - 2, :])))
    return max(r)

#%% Step 4 of Algorithm 1
def Sigma_q(Q_four_lams):
    """
    sample covariance matrix, prepare for resampling
    Paras:
    lams: (Q-1)-len list of four lam matrices (n * T-q * B)
    """
    sigma_q_s_max, sigma_q_s_mean = [], []
    Q = len(Q_four_lams) + 1
    q = 2
    for four_lams in Q_four_lams:  # for each q
        lam_RR, lam_II, lam_IR, lam_RI = four_lams  # (n * T-q * B) matrix

        lam = concatenate([lam_RR - lam_II, lam_RI + lam_IR],
                          2)  # into (n * T-q * 2B)
        N, T_q, BB = lam.shape
        sigma_q = np.zeros((BB, BB))
        for i in range(N):
            # aggregate across T with the .dot()
            sigma_q += lam[i].T.dot(lam[i])
        sigma_q_s_max.append(sigma_q / (N * T_q))
        q += 1
    return sigma_q_s_max


#%% Step 5 of Algorithm 1
def bootstrap_p_value(Q_Sigma_q, rep_times, test_stat=0):
    """
    resampling to get cv/p-values
    """
    BB = Q_Sigma_q[0].shape[0]
    Q = len(Q_Sigma_q) + 1
    Sigma_q_squares = [sqrtm(a) for a in Q_Sigma_q]

    def one_time(seed):
        rseed(seed); npseed(seed)
        Z = randn(BB, Q - 1)
        r = []
        for q in range(Q - 1):
            z = Z[:, q]
            r.append(max(Sigma_q_squares[q].dot(z)))
        return max(r)
    # generate boostrapped test stats
    r = rep_seeds(one_time, rep_times)
    p = p_value(test_stat, r)
    return p
