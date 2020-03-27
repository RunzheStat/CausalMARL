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
    for pattern_seed in range(4):
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

# --------------------------------------
#  [lam, w_hidden, pattern_seed, sd_D, w_spatial, l, T, random_target] =  [0.01, 30, 0, 1, 1, 5, 672, False]
# 13:37, 03/10; num of cores:36
# 0 1 1 0 0

# 1 0 0 0 0

# 1 1 0 0 1

# 0 1 1 0 1

# 0 0 1 1 1

# MC-based mean [average reward] and its std: [6.83  0.016]
# DR, IS, Susan, DR_NS
#  bias: [0.005 0.007 0.004 0.004]
#  std: [0.026 0.027 0.019 0.026]
#  MSE: [0.026 0.028 0.019 0.026]
# time spent until now: 298.6 mins

