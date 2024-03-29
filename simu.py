# export openblas_num_threads=1; export OMP_NUM_THREADS=1; python simu.py

from utils import * 
from weight import * 
from simu_funs import *  
from main import *
a = now()
##########################################################################################################################################################

rep_times = 100
region_parallel = False
full_parallel = False
pattern_seed = 2
sd_u_O = 25
w_O = .5
w_A = 1.5
u_D_range = [80] 
thre_range = [100, 101, 105, 110]

penalty_range = [[3e-4, 1e-4, 5e-5], [3e-4, 1e-4, 5e-5]]
with_NO_MARL = True
with_IS = True
with_MF = True
n_layer = 3
max_iteration = 1001
Learning_rate = 5e-4
w_hidden = 30
batch_size = 32


##########################################################################################################################################################

aim = "final_sd"

sd_R_range = [0, 5, 10, 15, 20, 25, 30]
day_range = [7]
print("[thre_range, sd_R_range, day_range, penalty_range]: ", [thre_range, sd_R_range, day_range, penalty_range])
results, res_real = [], []

for u_D in u_D_range:
    for sd_R in sd_R_range:
        for day in day_range:
            print_setting = DASH + "[pattern_seed, day, sd_R, u_D] = " + str([pattern_seed, day, sd_R, u_D]) + "\n"
            print(print_setting)
            N_targets = simu(pattern_seed = pattern_seed, l = 5, T = day * 48, t_func = None, # Setting - general
                     thre_range = thre_range, u_D = u_D, # Setting - general
                     sd_D = None, sd_R = sd_R, sd_O = None, sd_u_O = sd_u_O, # Setting - noise
                      w_A = w_A, w_O = w_O,  # Setting - spatial
                      # fixed
                      n_cores = n_cores, OPE_rep_times = rep_times, inner_parallel = region_parallel, full_parallel = full_parallel, # Parallel
                      CV_QV = True, penalty = penalty_range,
                      penalty_NMF = [[1e-3], [1e-3]], # Q-V hyperparameters;  penalty_NMF = [[1e-3, 1e-4, 1e-5], [1e-3, 1e-4, 1e-5]], 
                      w_hidden = w_hidden, n_layer = n_layer,  # NN hyperparameters
                      batch_size = batch_size, max_iteration = max_iteration, Learning_rate = Learning_rate,  # NN training
                      dim_S_plus_Ts = 3 + 3, epsilon = 1e-6, # Fixed
                      with_MF = with_MF, with_NO_MARL = with_NO_MARL, with_IS = with_IS)
            # N_targets: a list (len-N_target) of list of [bias, std, MSE, ...] (each of these measures is a vector)
            res_real.append(arr([a[2] for a in N_targets])) # N_targets * N_method for MSE
            for i in range(len(res_real)):
                print(res_real[i])
                print("\n")
            print("time spent until now:", np.round((now() - a)/60, 1), "mins", "\n")
            results.append(N_targets)
with open(aim + ".txt", "wb") as fp:
    pickle.dump(results, fp)
    
##########################################################################################################################################################

aim = "final_T"
sd_R_range = [15]
day_range = [2, 3, 4, 5, 6, 7, 8]
print("[thre_range, sd_R_range, day_range, penalty_range]: ", [thre_range, sd_R_range, day_range, penalty_range])
results, res_real = [], []

for u_D in u_D_range:
    for sd_R in sd_R_range:
        for day in day_range:
            print_setting = DASH + "[pattern_seed, day, sd_R, u_D] = " + str([pattern_seed, day, sd_R, u_D]) + "\n"
            print(print_setting)
            N_targets = simu(pattern_seed = pattern_seed, l = 5, T = day * 48, t_func = None, # Setting - general
                     thre_range = thre_range, u_D = u_D, # Setting - general
                     sd_D = None, sd_R = sd_R, sd_O = None, sd_u_O = sd_u_O, # Setting - noise
                      w_A = w_A, w_O = w_O,  # Setting - spatial
                      # fixed
                      n_cores = n_cores, OPE_rep_times = rep_times, inner_parallel = region_parallel, full_parallel = full_parallel, # Parallel
                      CV_QV = True, penalty = penalty_range,
                      penalty_NMF = [[1e-3], [1e-3]], # Q-V hyperparameters;  penalty_NMF = [[1e-3, 1e-4, 1e-5], [1e-3, 1e-4, 1e-5]], 
                      w_hidden = w_hidden, n_layer = n_layer,  # NN hyperparameters
                      batch_size = batch_size, max_iteration = max_iteration, Learning_rate = Learning_rate,  # NN training
                      dim_S_plus_Ts = 3 + 3, epsilon = 1e-6, # Fixed
                      with_MF = with_MF, with_NO_MARL = with_NO_MARL, with_IS = with_IS)
            # N_targets: a list (len-N_target) of list of [bias, std, MSE, ...] (each of these measures is a vector)
            res_real.append(arr([a[2] for a in N_targets])) # N_targets * N_method for MSE
            for i in range(len(res_real)):
                print(res_real[i])
                print("\n")
            print("time spent until now:", np.round((now() - a)/60, 1), "mins", "\n")
            results.append(N_targets)
with open(aim + ".txt", "wb") as fp:
    pickle.dump(results, fp)
    
##########################################################################################################################################################

# with open("0330.txt", "rb") as fp:
#     b = pickle.load(fp)
# printR("final sd_R trend for" + str(sd_R_range) + " the same \n")
