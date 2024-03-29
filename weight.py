"""
This file is required for the function "getWeight", which is used to get density ratio for value estimation.
The majority is adapted from the source code maintained on on Github for the paper "Breaking the Curse of Horizon: Infinite-Horizon Off-Policy Estimation".
    Date: 02/27/2020.
    URL: https://github.com/zt95/infinite-horizon-off-policy-estimation/blob/master/sumo/Density_ratio_continuous.py#L48
"""
##########################################################################################################################################################
import numpy as np
from utils import *
import tensorflow as tf # tf.__version__ = 1.13.2
from time import sleep
import sys
from scipy.stats import binom

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("!!", e)
tf.keras.backend.set_floatx('float64')


import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


####### Mute Warnings #############################################################################
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
# logging.getLogger('tensorflow').disabled = True
if type(tf.contrib) != type(tf): tf.contrib._warning = None
import warnings
warnings.filterwarnings('ignore')
##########################################################################################################################################################


class Density_Ratio_kernel(object):
    """ Modification
    # S -> (S, T_s)
    # A -> (A, T_a)
    # NN: s -> density. obs_dim here = dim_S + dim_{T_s}
    """
    def __init__(self, obs_dim, w_hidden, Learning_rate, reg_weight, n_layer = 2, gpu_number = 0):
        """ place holder. NN architecture
        """
#         tf.random.set_seed(0)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print("!!gpus:", gpus)
        self.gpu_number = 100
        tf.config.set_soft_device_placement(True)

        with tf.device('/gpu:' + str(self.gpu_number)):
            #tf.debugging.set_log_device_placement(True)
            print("self.gpu_number = ", self.gpu_number)

            self.reg_weight = reg_weight
            self.n_layer = n_layer

            self.state = tf.placeholder(tf.float32, [None, obs_dim])
            self.med_dist = tf.placeholder(tf.float32, [])
            self.next_state = tf.placeholder(tf.float32, [None, obs_dim])

            self.state2 = tf.placeholder(tf.float32, [None, obs_dim])
            self.next_state2 = tf.placeholder(tf.float32, [None, obs_dim])
            self.policy_ratio = tf.placeholder(tf.float32, [None])
            self.policy_ratio2 = tf.placeholder(tf.float32, [None])

            """ density ratio for (s_i, s'_i) and (s_j, s'_j)
            """
            ## get densities
            w1 = self.state_to_w(self.state, obs_dim, w_hidden)
            w_next = self.state_to_w(self.next_state, obs_dim, w_hidden)
            w2 = self.state_to_w(self.state2, obs_dim, w_hidden)
            w_next2 = self.state_to_w(self.next_state2, obs_dim, w_hidden)
            ## normalization for weight means
            norm_w = tf.reduce_mean(w1)
            norm_w_next = tf.reduce_mean(w_next)
            norm_w2 = tf.reduce_mean(w2)
            norm_w_next2 = tf.reduce_mean(w_next2)

            self.output = w1 # the state density ratio: what we need

            """ calculate loss function
            """
            ## 1. two Delta 

            x = w1 * self.policy_ratio / norm_w - w_next / norm_w_next
            x2 = w2 * self.policy_ratio2 / norm_w2 - w_next2 / norm_w_next2

            ## 2. K(.,.)
            diff_xx = tf.expand_dims(self.next_state, 1) - tf.expand_dims(self.next_state2, 0) # as a matrix
            K_xx = tf.exp(-tf.reduce_sum(tf.square(diff_xx), axis = -1)/(2.0*self.med_dist*self.med_dist))
            norm_K = tf.reduce_sum(K_xx)

            ## 3. final step in the paper: vec * matrix * vec = scalar = \hat{D(w)} (w/o /|M| )
            loss_xx = tf.matmul(tf.matmul(tf.expand_dims(x, 0), K_xx),tf.expand_dims(x2, 1))

            ### Transform as NN loss
            # The normalization part in that paper
            self.loss = tf.squeeze(loss_xx) #/(norm_w*norm_w2*norm_K)
            self.reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'w'))
            self.train_op = tf.train.AdamOptimizer(Learning_rate).minimize(self.loss +reg_weight * self.reg_loss) 

            # Debug
            self.debug1 = tf.reduce_mean(w1)
            self.debug2 = tf.reduce_mean(w_next)

            # Initializaiton of the session
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    def reset(self):
#         with tf.device('/gpu:' + str(self.gpu_number)):
        self.sess.run(tf.global_variables_initializer())

    def close_Session(self):
#         with tf.device('/gpu:' + str(self.gpu_number)):
        tf.reset_default_graph()
        self.sess.close()

    """ construct the NN
    """
    def state_to_w(self, state, obs_dim, hidden_dim_dr):# state_to_w_tl 
        with tf.device('/gpu:' + str(self.gpu_number)):
            if self.n_layer == 2:
                with tf.variable_scope('w', reuse=tf.AUTO_REUSE):
                    # First layer
                    W1 = tf.get_variable('W1', initializer = tf.random_normal(shape = [obs_dim, hidden_dim_dr])) 
                    b1 = tf.get_variable('b1', initializer = tf.zeros([hidden_dim_dr]))
                    z1 = tf.matmul(state, W1) + b1
                    mean_z1, var_z1 = tf.nn.moments(z1, [0])
                    scale_z1 = tf.get_variable('scale_z1', initializer = tf.ones([hidden_dim_dr]))
                    beta_z1 = tf.get_variable('beta_z1', initializer = tf.zeros([hidden_dim_dr]))
                    l1 = tf.nn.leaky_relu(tf.nn.batch_normalization(z1, mean_z1, var_z1, beta_z1, scale_z1, 1e-10))


                    # Second layer
                    W2 = tf.get_variable('W2', initializer = 0.01 * tf.random_normal(shape = [hidden_dim_dr, 1]), regularizer = tf.contrib.layers.l2_regularizer(1.))
                    b2 = tf.get_variable('b2', initializer = tf.zeros([1]), regularizer = tf.contrib.layers.l2_regularizer(1.))
                    z2 = tf.matmul(l1, W2) + b2

                    return tf.log(1+tf.exp(tf.squeeze(z2))) # = log(1 + e^z2)

            elif self.n_layer == 3:
                with tf.variable_scope('w', reuse=tf.AUTO_REUSE):
                    # First layer
                    W1 = tf.get_variable('W1', initializer = tf.random_normal(shape = [obs_dim, hidden_dim_dr]))

                    b1 = tf.get_variable('b1', initializer = tf.zeros([hidden_dim_dr]))#, regularizer = tf.contrib.layers.l2_regularizer(1.))
                    z1 = tf.matmul(state, W1) + b1
                    mean_z1, var_z1 = tf.nn.moments(z1, [0])
                    scale_z1 = tf.get_variable('scale_z1', initializer = tf.ones([hidden_dim_dr]))
                    beta_z1 = tf.get_variable('beta_z1', initializer = tf.zeros([hidden_dim_dr]))
                    l1 = tf.nn.leaky_relu(tf.nn.batch_normalization(z1, mean_z1, var_z1, beta_z1, scale_z1, 1e-10))

                    # Second layer
                    W2 = tf.get_variable('W2', initializer = tf.random_normal(shape = [hidden_dim_dr, hidden_dim_dr])) #, regularizer = tf.contrib.layers.l2_regularizer(1.))
                    b2 = tf.get_variable('b2', initializer = tf.zeros([hidden_dim_dr])) #, regularizer = tf.contrib.layers.l2_regularizer(1.))
                    z2 = tf.matmul(l1, W2) + b2
                    mean_z2, var_z2 = tf.nn.moments(z2, [0])
                    scale_z2 = tf.get_variable('scale_z2', initializer = tf.ones([hidden_dim_dr]))
                    beta_z2 = tf.get_variable('beta_z2', initializer = tf.zeros([hidden_dim_dr]))
                    l2 = tf.nn.leaky_relu(tf.nn.batch_normalization(z2, mean_z2, var_z2, beta_z2, scale_z2, 1e-10))

                    # Third layer
                    # regularization parameters here!!!
                    W3 = tf.get_variable('W3', initializer = 0.01 * tf.random_normal(shape = [hidden_dim_dr, 1]), regularizer = tf.contrib.layers.l2_regularizer(1.))
                    b3 = tf.get_variable('b3', initializer = tf.zeros([1]), regularizer = tf.contrib.layers.l2_regularizer(1.))
                    z3 = tf.matmul(l2, W3) + b3
                    return tf.log(1 + tf.exp(tf.squeeze(z3))) # := log(1 + e^z2)

    def get_density_ratio(self, states):
        with tf.device('/gpu:' + str(self.gpu_number)):
            return self.sess.run(self.output, feed_dict = {
                self.state : states
                })

    def train(self, SASR, policy0, policy1, print_flag, 
              batch_size, max_iteration,
              test_num = 0, 
              epsilon = 1e-3, only_state = False, 
              n_neigh = 8, spatial = True, mean_field = True):
        """ Train the NN to get theta^*
        Input:
            policy1(state, action) is its probability
        """
        ########################### Preparing data ###########################
        with tf.device('/gpu:' + str(self.gpu_number)):
            npseed(0)
            S = []
            SN = []
            PI1 = []
            PI0 = []
            REW = []
            for sasr in SASR:
                for state, action, next_state, reward, pi_Sit in sasr:
                    if spatial: # p_{a|s} needs to consider the neigh (only for our behaviour cases)
                        Ta_tl = action[1]
                        A_tl = action[0]
                        if mean_field:
                            pi1 = int(int(state[0]) == Ta_tl and pi_Sit == A_tl) # P_tp(a = obs_action | S)
                            PI1.append(pi1) # the target policy is deterministic
                            pi0 = den_b_disc(Ta_tl, n_neigh) * 0.5
                            PI0.append(pi0)
                            if pi0 == 0:
                                print([Ta_tl, n_neigh])
                        else:# not mean-field
                            pi1 = int(np.array_equal(pi_Sit[1], Ta_tl) and pi_Sit[0] == A_tl)
                            PI1.append(pi1) # the target policy is deterministic
                            pi0 = 0.5 ** (n_neigh + 1)
                            PI0.append(pi0)                            
                    else:#                     PI1.append(1)
                        pi1 = int(pi_Sit == action)
                        PI1.append(pi1)
                        PI0.append(0.5) # purely random

                    S.append(state)
                    SN.append(next_state)
                    REW.append(reward) # not immediately required in DR
            if print_flag is True:
                print("spatial is", spatial, ": #{PI1 = 1} = ", sum(PI1), "in ",  len(PI1))        
                print("PI0:", np.mean(PI0), np.std(PI0))
            ## Normalization
            S = np.array(S)
            SN = np.array(SN)
            S_max = np.max(S, axis = 0)
            S_min = np.min(S, axis = 0)

            if spatial and mean_field:
                S[:, 1:] = (S[:, 1:] - S_min[1:])/(S_max[1:] - S_min[1:])
                SN[:, 1:] = (SN[:, 1:] - S_min[1:])/(S_max[1:] - S_min[1:])
            else:
                S = (S - S_min)/(S_max - S_min)
                SN = (SN - S_min)/(S_max - S_min)

            ## Keep the original data for getting fitting results
            S_whole = np.array(S).copy()
            PI1_whole = np.array(PI1).copy()
            PI0_whole = np.array(PI0).copy()

            ## Training data
            S = np.array(S[test_num:])
            SN = np.array(SN[test_num:])
            PI1 = np.array(PI1[test_num:])
            PI0 = np.array(PI0[test_num:])
            REW = np.array(REW[test_num:])
            N = S.shape[0]

            ## Testing data
            if test_num > 0:
                S_test = np.array(S[:test_num])
                SN_test = np.array(SN[:test_num])
                PI1_test = np.array(PI1[:test_num])
                PI0_test = np.array(PI0[:test_num])

            #################################################################################

            ## Get the med_dist for hyperparameter in RKHS
            subsamples = np.random.choice(N, 1000)
            s = S[subsamples]
            med_dist = np.median(np.sqrt(np.sum(np.square(s[None, :, :] - s[:, None, :]), axis = -1)))

            for i in range(max_iteration):
                ## monitor the performance on test set
                if test_num > 0 and i % 100 == 0:
                    subsamples = np.array(range(test_num))
                    # subsamples = np.random.choice(test_num, batch_size)
                    s_test = S_test[subsamples]
                    sn_test = SN_test[subsamples]
                    policy_ratio_test = (PI1_test[subsamples] + epsilon)/(PI0_test[subsamples] + epsilon)

                    # subsamples = np.random.choice(test_num, batch_size)
                    s_test2 = S_test[subsamples]
                    sn_test2 = SN_test[subsamples]
                    policy_ratio_test2 = (PI1_test[subsamples] + epsilon)/(PI0_test[subsamples] + epsilon)

                    test_loss, reg_loss, norm_w, norm_w_next = self.sess.run([self.loss, self.reg_loss, self.debug1, self.debug2], feed_dict = {
                        self.med_dist: med_dist,
                        self.state: s_test,
                        self.next_state: sn_test,
                        self.policy_ratio: policy_ratio_test,
                        self.state2: s_test2,
                        self.next_state2: sn_test2,
                        self.policy_ratio2: policy_ratio_test2
                        })
                    print('----Iteration = {}-----'.format(i))
                    printR("Testing error = {:.5}".format(test_loss))
                    print('Regularization loss = {}'.format(reg_loss))
                    print('Norm_w = {:.5}'.format(norm_w)," || ", 'Norm_w_next = {:.5}'.format(norm_w_next))
                    Density_ratio = self.get_density_ratio(S)
                    T = Density_ratio * PI1 / PI0 
                    print('mean_reward_estimate = {:.5}'.format(np.sum(T*REW)/np.sum(T)))
                    sys.stdout.flush()
                    # epsilon *= 0.9 # for stability

                ## Training: pick subsample and run the algorithm 

                subsamples = np.random.choice(N, batch_size) 

                s = S[subsamples]
                sn = SN[subsamples]
                policy_ratio = (PI1[subsamples] + epsilon)/(PI0[subsamples] + epsilon)
                s2 = S[subsamples]
                sn2 = SN[subsamples]
                policy_ratio2 = (PI1[subsamples] + epsilon)/(PI0[subsamples] + epsilon)
                _, train_loss = self.sess.run([self.train_op, self.loss], feed_dict = {
                    self.med_dist: med_dist,
                    self.state: s,
                    self.next_state: sn,
                    self.policy_ratio: policy_ratio,
                    self.state2 : s2,
                    self.next_state2: sn2,
                    self.policy_ratio2: policy_ratio2
                    })


            state_ratio = self.get_density_ratio(S_whole)
            action_ratio = PI1_whole / PI0_whole

            ratio_whole = state_ratio *  action_ratio

            if test_num > 0:
                print(DASH)
            if only_state:
                return state_ratio
            else:
                return ratio_whole # v_i in that paper
