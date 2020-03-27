"""
This file is XXX. 
The majority is adapted from the source code of the paper "Breaking the Curse of Horizon: Infinite-Horizon Off-Policy Estimation" on Github.
    Date: 02/27/2020.
    URL: https://github.com/zt95/infinite-horizon-off-policy-estimation/blob/master/sumo/Density_ratio_continuous.py#L48
"""
#######
import numpy as np
from _uti_basic import *
import tensorflow as tf
from time import sleep
import sys
from scipy.stats import binom


####### Mute Warnings
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
# logging.getLogger('tensorflow').disabled = True
if type(tf.contrib) != type(tf): tf.contrib._warning = None
import warnings
warnings.filterwarnings('ignore')
#######

state_batch_dim = 3 # the original is 4. why?

class Density_Ratio_kernel(object):
    """ Modification
    # S -> (S, T_s)
    # A -> (A, T_a)
    # NN: s -> density. obs_dim here = dim_S + dim_{T_s}
    """
    def __init__(self, obs_dim, w_hidden, Learning_rate, reg_weight, n_layer = 2):
        """ place holder. NN architecture
        """
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
        ## 1. two Delta [with three kinds of normalization]
        
#         x = w1 * self.policy_ratio - w_next
#         x2 = w2 * self.policy_ratio2 - w_next2
        x = w1 * self.policy_ratio / norm_w - w_next / norm_w_next
        x2 = w2 * self.policy_ratio2 / norm_w2 - w_next2 / norm_w_next2
        # norm_w_beta = tf.reduce_mean(w * self.policy_ratio) # what is the application
        # norm_w_beta2 = tf.reduce_mean(w2 * self.policy_ratio2)
#         x = w * self.policy_ratio / norm_w_beta - w_next / norm_w
#         x2 = w2 * self.policy_ratio2 / norm_w_beta2 - w_next2 / norm_w2
        
        ## 2. K(.,.)
        diff_xx = tf.expand_dims(self.next_state, 1) - tf.expand_dims(self.next_state2, 0) # as a matrix
        K_xx = tf.exp(-tf.reduce_sum(tf.square(diff_xx), axis = -1)/(2.0*self.med_dist*self.med_dist))
        norm_K = tf.reduce_sum(K_xx)

        ## 3. final step in the paper: vec * matrix * vec = scalar = \hat{D(w)} (w/o /|M| )
        loss_xx = tf.matmul(tf.matmul(tf.expand_dims(x, 0), K_xx),tf.expand_dims(x2, 1))
        
        ### Transform as NN loss
        # The normalization part in that paper
        self.loss = tf.squeeze(loss_xx) #/(norm_w*norm_w2*norm_K)
#         self.loss = tf.squeeze(loss_xx)/ norm_w
#         self.loss = tf.squeeze(loss_xx) / norm_K
#         self.reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'w'))
        self.train_op = tf.train.AdamOptimizer(Learning_rate).minimize(self.loss) # mute the later part?+reg_weight * self.reg_loss # not mentioned in the paper

        # Debug: what can we find?
        self.debug1 = tf.reduce_mean(w1)
        self.debug2 = tf.reduce_mean(w_next)

        # Initializaiton of the session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def reset(self):
        self.sess.run(tf.global_variables_initializer())

    def close_Session(self):
        tf.reset_default_graph()
        self.sess.close()

    """ construct the NN
    """
    def state_to_w(self, state, obs_dim, hidden_dim_dr):# state_to_w_tl 
        if self.n_layer == 2:
            with tf.variable_scope('w', reuse=tf.AUTO_REUSE):
                # First layer
                W1 = tf.get_variable('W1', initializer = tf.random_normal(shape = [obs_dim, hidden_dim_dr])) #, regularizer = tf.contrib.layers.l2_regularizer(1.))
                b1 = tf.get_variable('b1', initializer = tf.zeros([hidden_dim_dr]))
                #, regularizer = tf.contrib.layers.l2_regularizer(1.))
                z1 = tf.matmul(state, W1) + b1
                mean_z1, var_z1 = tf.nn.moments(z1, [0])
                scale_z1 = tf.get_variable('scale_z1', initializer = tf.ones([hidden_dim_dr]))
                beta_z1 = tf.get_variable('beta_z1', initializer = tf.zeros([hidden_dim_dr]))
                l1 = tf.nn.leaky_relu(tf.nn.batch_normalization(z1, mean_z1, var_z1, beta_z1, scale_z1, 1e-10))


                # Second layer
                W2 = tf.get_variable('W2', initializer = 0.01 * tf.random_normal(shape = [hidden_dim_dr, 1]), regularizer = tf.contrib.layers.l2_regularizer(1.))
                b2 = tf.get_variable('b2', initializer = tf.zeros([1]), regularizer = tf.contrib.layers.l2_regularizer(1.))
                z2 = tf.matmul(l1, W2) + b2
                #return tf.exp(tf.squeeze(z2))
                #mean_z2, var_z2 = tf.nn.moments(z2, [0])
                #scale_z2 = tf.get_variable('scale_z2', initializer = tf.ones([1]))
                #beta_z2 = tf.get_variable('beta_z2', initializer = tf.zeros([1]))
                #l2 = tf.nn.batch_normalization(z2, mean_z2, var_z2, beta_z2, scale_z2, 1e-10)
                return tf.log(1+tf.exp(tf.squeeze(z2))) # = log(1 + e^z2)
        
        elif self.n_layer == 3:
            with tf.variable_scope('w', reuse=tf.AUTO_REUSE):
                # First layer
                W1 = tf.get_variable('W1', initializer = tf.random_normal(shape = [obs_dim, hidden_dim_dr])) #, regularizer = tf.contrib.layers.l2_regularizer(1.))
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
        return self.sess.run(self.output, feed_dict = {
            self.state : states
            })

    def train(self, SASR, policy0, policy1, 
              batch_size, max_iteration,
              test_num, epsilon = 1e-3, only_state = False, n_neigh = 8, spatial = True):
        """ Train the NN to get theta^*
        Input:
            policy1(state, action) is its probability
            
        """
        ########################### Preparing data ###########################
        S = []
        SN = []
        PI1 = []
        PI0 = []
        REW = []
        for sasr in SASR:
            for state, action, next_state, reward in sasr:
#                 PI1.append(policy1(state, action))
                PI1.append(1) # fixed
                if spatial: # p_{a|s} needs to consider the neigh (only for our behaviour cases)
                    PI0.append(0.5 * binom.pmf(action, n_neigh, 0.5))# policy0(state, action)
                else:
                    PI0.append(0.5) # purely random
                S.append(state)
                SN.append(next_state)
                REW.append(reward) # not immediately required in DR
        
        ## Normalization
        S = np.array(S)
        S_max = np.max(S, axis = 0)
        S_min = np.min(S, axis = 0)
        S = (S - S_min)/(S_max - S_min)
        SN = (np.array(SN) - S_min)/(S_max - S_min)
        
        
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

                test_loss, norm_w, norm_w_next = self.sess.run([self.loss, self.debug1, self.debug2], feed_dict = {
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
                print('Norm_w = {:.5}'.format(norm_w)," || ", 'Norm_w_next = {:.5}'.format(norm_w_next))
                Density_ratio = self.get_density_ratio(S)
                T = Density_ratio * PI1 / PI0 #???
                print('mean_reward_estimate = {:.5}'.format(np.sum(T*REW)/np.sum(T)))
                sys.stdout.flush()
                # epsilon *= 0.9 # for stability
            
            ## Training: pick subsample and run the algorithm 
            # vectorization -> thus s = s2 
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
#             if test_num == 0 and i % 10 == 0: # check if we can overfit it
#                 print(i, " training losses:", train_loss)
                
        state_ratio = self.get_density_ratio(S_whole)
        action_ratio = PI1_whole / PI0_whole
        ratio_whole = state_ratio *  action_ratio
        
#         print(np.mean(state_ratio), np.std(state_ratio))
#         print(np.mean(action_ratio), np.std(action_ratio))
#         print(np.mean(ratio_whole), np.std(ratio_whole), "\n")

        if test_num > 0:
            print(DASH)
        if only_state:
            return state_ratio
        else:
            return ratio_whole # v_i in that paper
