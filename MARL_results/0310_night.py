# from _uti_basic import *
from _utility import *
from weight import *
from simu_funs import *
from main import *
os.environ["OMP_NUM_THREADS"] = "1"

l = 5
T = 14 * 48
rep_times = int(n_cores)
a = now()
lam = 0.01
w_hidden = 30
sd_D = 1
w_spatial = 1
for random_target in [False, True]:
    for pattern_seed in range(2, 4):
        print(DASH, "[lam, w_hidden, pattern_seed, sd_D, w_spatial, l, T, random_target] = ", [lam, w_hidden, pattern_seed, sd_D, w_spatial, l, T, random_target])
        r = simu(pattern_seed = pattern_seed, l = l, T = T, sd_D = sd_D, sd_R = 0, random_target = random_target, # Setting - general
                 simple_grid_neigh = False, w_A = w_spatial, w_O = w_spatial,  # Setting - spatial
                  n_cores = n_cores, OPE_rep_times = rep_times, inner_parallel = False, # Parallel
                  penalty = [[lam], [lam]], # Q-V hyperparameters
                  w_hidden = w_hidden, n_layer = 2,  # NN hyperparameters
                  batch_size = 32, max_iteration = 1001, Learning_rate = 1e-3, test_num = 0,  # NN training
                  dim_S_plus_Ts = 3 + 3, epsilon = 1e-6 # Fixed
                  )
        print( "time spent until now:", np.round((now() - a)/60, 1), "mins", "\n")


# --------------------------------------                                                                                                                                                                                                                              [5/1971]
#  [lam, w_hidden, pattern_seed, sd_D, w_spatial, l, T] =  [0.01, 30, 0, 1, 1, 5, 672]                                                                                                                                                                                        
# 22:46, 03/10; num of cores:16                                                                                                                                                                                                                                               
# 0 1 1 0 0                                                                                                                                                                                                                                                                   
                                                                                                                                                                                                                                                                            
# 1 0 0 0 0                                                                                                                                                                                                                                                                   

# 1 1 0 0 1

# 0 1 1 0 1

# 0 0 1 1 1

# MC-based mean [average reward] and its std: [6.83  0.016]
# DR, IS, Susan, DR_NS
#  bias: [0.004 0.005 0.    0.003]
#  std: [0.019 0.019 0.017 0.018]
#  MSE: [0.019 0.02  0.017 0.018]
# time spent until now: 6.3 mins


# --------------------------------------
#  [lam, w_hidden, pattern_seed, sd_D, w_spatial, l, T] =  [0.01, 30, 1, 1, 1, 5, 672]
# 22:52, 03/10; num of cores:16
# 0 0 1 1 1

# 0 1 0 0 0

# 1 0 0 1 1

# 0 0 1 1 1

# 1 1 0 0 0

# MC-based mean [average reward] and its std: [6.83  0.016]
# DR, IS, Susan, DR_NS
#  bias: [0.005 0.006 0.    0.004]
#  std: [0.022 0.021 0.018 0.024]
#  MSE: [0.023 0.022 0.018 0.024]
# time spent until now: 12.6 mins



#  [lam, w_hidden, pattern_seed, sd_D, w_spatial, l, T, random_target] =  [0.01, 30, 2, 1, 1, 5, 672, False]                                                                                                                                                                  
# MC-based mean [average reward] and its std: [6.811 0.016]
# DR, IS, Susan, DR_NS
#  bias: [0.01  0.009 0.019 0.012]
#  std: [0.013 0.013 0.017 0.016]
#  MSE: [0.016 0.016 0.025 0.02 ]
# time spent until now: 6.0 mins


#  [lam, w_hidden, pattern_seed, sd_D, w_spatial, l, T, random_target] =  [0.01, 30, 3, 1, 1, 5, 672, False]
# MC-based mean [average reward] and its std: [6.663 0.015]                                                                                                                                                                                                                   
# DR, IS, Susan, DR_NS                                                                                                                                                                                                                                                        
#  bias: [0.16  0.158 0.167 0.159]                                                                                                                                                                                                                                            
#  std: [0.019 0.019 0.017 0.021]                                                                                                                                                                                                                                             
#  MSE: [0.161 0.159 0.168 0.16 ]
# time spent until now: 11.9 mins


------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
#  [lam, w_hidden, pattern_seed, sd_D, w_spatial, l, T, random_target] =  [0.01, 30, 2, 1, 1, 5, 672, True]                                                                                                                                                                   

# MC-based mean [average reward] and its std: [6.658 0.016]                                                                                                                                                                                                                   
# DR, IS, Susan, DR_NS                                                                                                                                                                                                                                                        
#  bias: [0.165 0.163 0.172 0.163]                                                                                                                                                                                                                                            
#  std: [0.028 0.028 0.018 0.027]                                                                                                                                                                                                                                             
#  MSE: [0.167 0.165 0.173 0.165]   


# MC-based mean [average reward] and its std: [6.96  0.016]
# DR, IS, Susan, DR_NS 
#  bias: [0.136 0.138 0.129 0.137] 
#  std: [0.023 0.022 0.017 0.024] 
#  MSE: [0.138 0.14  0.13  0.139]
# time spent until now: 24.3 mins 
