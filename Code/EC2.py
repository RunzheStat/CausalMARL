from _utility import *
from weight import *
from simu_funs import *
from main import *
import os 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
a = now()
# export openblas_num_threads=1; export OMP_NUM_THREADS=1

# if n_cores <= 16:
#     rep_times = 48 
# elif n_cores >= 36:
rep_times = n_cores
# rep_times = 16

l = 5
T = 14 * 48
lam = 0.01

sd_R = 0
w_A = 1

path = "0327.txt"
file = open(path, 'w') 

shared_setting = "Basic setting:" + "[l, T, lam, sd_R, w_A] = " + str([l, T, lam, sd_R, w_A]) + "\n"
print(shared_setting)
print(shared_setting, file = file)

#             for T in [int(14 * 48)]: # , int(14 * 48 * 2) # int(14 * 48 / 2), 

for sd_D in [3, 1]:
    for w_O in [0.05, 0.01]:
        for pattern_seed in [None, 1, 2, 3, 4]:
            setting = DASH + "[pattern_seed, sd_D, w_O] = " + str([pattern_seed, sd_D, w_O]) + "\n"
            print(setting)
            print(setting, file = file)
            r = simu(pattern_seed = pattern_seed, l = l, T = T, time_dependent = False, # Setting - general
                     sd_D = sd_D, sd_R = sd_R, # Setting - noise
                      w_A = w_A, w_O = w_O,  # Setting - spatial
                      n_cores = n_cores, OPE_rep_times = rep_times, inner_parallel = False, # Parallel
                      penalty = [[lam], [lam]], # Q-V hyperparameters
                      w_hidden = 30, n_layer = 2,  # NN hyperparameters
                      batch_size = 32, max_iteration = 1001, Learning_rate = 1e-3,  # NN training
                      dim_S_plus_Ts = 3 + 3, epsilon = 1e-6, # Fixed
                      isValidation = False, test_num = 0, # test
                      file = file
                      )
            print("time spent until now:", np.round((now() - a)/60, 1), "mins", "\n")

file.close() 

