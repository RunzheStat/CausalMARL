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
sd_R = 2
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
