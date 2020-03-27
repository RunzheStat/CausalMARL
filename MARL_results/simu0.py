# from _uti_basic import *
from _utility import *
from weight import *
from simu_funs import *
from main import *
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# export openblas_num_threads=1
# export OMP_NUM_THREADS=1

l = 5
T = 14 * 48
rep_times = int(n_cores)
a = now()

w_hidden = 30
sd_D = 3
sd_R = 0
w_spatial = 1

lam = 0.01

for pattern_seed in range(4):
    print(DASH, "[lam, w_hidden, pattern_seed, sd_D, w_spatial, l, T] = ", [lam, w_hidden, pattern_seed, sd_D, w_spatial, l, T])
    r = simu(pattern_seed = pattern_seed, l = l, T = T, sd_D = sd_D, sd_R = sd_R, # Setting - general
             random_target = True, time_dependent = False, mean_reversion = False, print_flag = False, # Setting - general
             simple_grid_neigh = False, w_A = w_spatial, w_O = w_spatial,  # Setting - spatial
              n_cores = n_cores, OPE_rep_times = rep_times, inner_parallel = False, # Parallel
              penalty = [[lam], [lam]], # Q-V hyperparameters
              w_hidden = w_hidden, n_layer = 2,  # NN hyperparameters
              batch_size = 32, max_iteration = 1001, Learning_rate = 1e-3, test_num = 0,  # NN training
              dim_S_plus_Ts = 3 + 3, epsilon = 1e-6 # Fixed
              )
    print( "time spent until now:", np.round((now() - a)/60, 1), "mins", "\n")



# mean_O = 5

# --------------------------------------
#  [lam, w_hidden, pattern_seed, sd_D, w_spatial, l, T] =  [0.01, 30, 0, 3, 1, 5, 672]
# 16:17, 03/12; num of cores:36
# 1 1 1 1 0 

# 1 0 1 1 0 

# 1 1 1 1 0 

# 0 0 1 1 1 

# 1 1 0 1 0 

# MC-based mean [average reward] and its std: [2.097 0.168]
# DR, IS, Susan, DR_NS 
#  bias: [0.058 0.058 0.067 0.062] 
#  std: [0.163 0.162 0.167 0.164] 
#  MSE: [0.173 0.172 0.18  0.175]
# time spent until now: 7.7 mins 


# --------------------------------------
#  [lam, w_hidden, pattern_seed, sd_D, w_spatial, l, T] =  [0.01, 30, 1, 3, 1, 5, 672]
# 16:25, 03/12; num of cores:36
# 0 1 0 0 0 

# 0 0 0 0 1 

# 0 1 0 1 0 

# 1 0 1 0 0 

# 1 1 0 1 1 

# MC-based mean [average reward] and its std: [2.028 0.163]
# DR, IS, Susan, DR_NS 
#  bias: [0.127 0.127 0.136 0.131] 
#  std: [0.163 0.162 0.167 0.164] 
#  MSE: [0.207 0.206 0.215 0.21 ]
# time spent until now: 15.2 mins 


# --------------------------------------
#  [lam, w_hidden, pattern_seed, sd_D, w_spatial, l, T] =  [0.01, 30, 2, 3, 1, 5, 672]
# 16:33, 03/12; num of cores:36
# 0 0 1 0 0 

# 0 0 1 0 0 

# 1 1 0 1 0 

# 1 1 0 1 0 

# 1 0 0 0 0 

# MC-based mean [average reward] and its std: [1.963 0.129]
# DR, IS, Susan, DR_NS 
#  bias: [0.194 0.193 0.201 0.196] 
#  std: [0.163 0.162 0.167 0.164] 
#  MSE: [0.253 0.252 0.261 0.256]
# time spent until now: 22.7 mins 

