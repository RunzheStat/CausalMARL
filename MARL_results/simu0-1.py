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
w_spatial = 1

lam = 0.01

for pattern_seed in range(8):
    print(DASH, "[lam, w_hidden, pattern_seed, sd_D, w_spatial, l, T] = ", [lam, w_hidden, pattern_seed, sd_D, w_spatial, l, T])
    r = simu(pattern_seed = pattern_seed, l = l, T = T, sd_D = sd_D, sd_R = 0, # Setting - general
             random_target = True, time_dependent = False, mean_reversion = False, print_flag = False, # Setting - general
             simple_grid_neigh = False, w_A = w_spatial, w_O = w_spatial,  # Setting - spatial
              n_cores = n_cores, OPE_rep_times = rep_times, inner_parallel = False, # Parallel
              penalty = [[lam], [lam]], # Q-V hyperparameters
              w_hidden = w_hidden, n_layer = 2,  # NN hyperparameters
              batch_size = 32, max_iteration = 1001, Learning_rate = 1e-3, test_num = 0,  # NN training
              dim_S_plus_Ts = 3 + 3, epsilon = 1e-6 # Fixed
              )
    print( "time spent until now:", np.round((now() - a)/60, 1), "mins", "\n")


# --------------------------------------
#  [lam, w_hidden, pattern_seed, sd_D, w_spatial, l, T] =  [0.01, 30, 0, 3, 1, 5, 672]
# 10:30, 03/12; num of cores:36
# 1 1 1 1 0 

# 1 0 1 1 0 

# 1 1 1 1 0 

# 0 0 1 1 1 

# 1 1 0 1 0 

# MC-based mean [average reward] and its std: [5.458 0.13 ]
# DR, IS, Susan, DR_NS 
#  bias: [0.17  0.161 0.205 0.201] 
#  std: [0.177 0.179 0.174 0.175] 
#  MSE: [0.245 0.241 0.269 0.267]
# time spent until now: 7.6 mins 


# --------------------------------------
#  [lam, w_hidden, pattern_seed, sd_D, w_spatial, l, T] =  [0.01, 30, 1, 3, 1, 5, 672]
# 10:38, 03/12; num of cores:36
# 0 1 0 0 0 

# 0 0 0 0 1 

# 0 1 0 1 0 

# 1 0 1 0 0 

# 1 1 0 1 1 

# MC-based mean [average reward] and its std: [5.392 0.172]
# DR, IS, Susan, DR_NS 
#  bias: [0.243 0.235 0.271 0.274] 
#  std: [0.18  0.182 0.174 0.175] 
#  MSE: [0.302 0.297 0.322 0.325]
# time spent until now: 15.1 mins 


# --------------------------------------
#  [lam, w_hidden, pattern_seed, sd_D, w_spatial, l, T] =  [0.01, 30, 2, 3, 1, 5, 672]
# 10:46, 03/12; num of cores:36
# 0 0 1 0 0 

# 0 0 1 0 0 

# 1 1 0 1 0 

# 1 1 0 1 0 

# 1 0 0 0 0 

# MC-based mean [average reward] and its std: [5.086 0.114]
# DR, IS, Susan, DR_NS 
#  bias: [0.547 0.537 0.577 0.576] 
#  std: [0.177 0.179 0.174 0.176] 
#  MSE: [0.575 0.566 0.603 0.602]
# time spent until now: 22.7 mins 

