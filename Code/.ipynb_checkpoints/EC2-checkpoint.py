##########################################################################################################################################################
from _utility import *
from weight import *
from simu_funs import *
from main import *
a = now()
# export openblas_num_threads=1; export OMP_NUM_THREADS=1

##########################################################################################################################################################
# rep_times = n_cores
rep_times = 2

l = 5
# T = 14 * 48
lam = 0.01

sd_R = 0
# w_A = 1
w_O = 0.05
inner_parallel = True
dynamics = "new"

# sd_D = 1
path = "0328.txt"
##########################################################################################################################################################
file = open(path, 'w') 
shared_setting = "Basic setting:" + "[l, lam, sd_R] = " + str([l, lam, sd_R]) + "\n"
print(shared_setting)
print(shared_setting, file = file)

##########################################################################################################################################################
#             for T in [int(14 * 48)]: # , int(14 * 48 * 2) # int(14 * 48 / 2), 
settings = []
for sd_D in [1, 3]:
    for w_A in [1, 2]:
        for T in [14 * 24, 14 * 48]:
            for pattern_seed in [None, 1, 2, 3, 4, 5, 6]:
                settings.append([pattern_seed, sd_D, w_A, T])
##########################################################################################################################################################
for setting in settings:
    pattern_seed, sd_D, w_A, T = setting
    print_setting = DASH + "[pattern_seed, sd_D, w_A, T] = " + str([pattern_seed, sd_D, w_A, T]) + "\n"
    print(print_setting)
    print(print_setting, file = file)
    r = simu(pattern_seed = pattern_seed, l = l, T = T, time_dependent = False, dynamics = dynamics, # Setting - general
             sd_D = sd_D, sd_R = sd_R, # Setting - noise
              w_A = w_A, w_O = w_O,  # Setting - spatial
              # fixed
              n_cores = n_cores, OPE_rep_times = rep_times, inner_parallel = inner_parallel, # Parallel
              penalty = [[lam], [lam]], # Q-V hyperparameters
              w_hidden = 30, n_layer = 2,  # NN hyperparameters
              batch_size = 32, max_iteration = 1001, Learning_rate = 1e-3,  # NN training
              dim_S_plus_Ts = 3 + 3, epsilon = 1e-6, # Fixed
              isValidation = False, test_num = 0, # test
              file = file
              )
    print("time spent until now:", np.round((now() - a)/60, 1), "mins", "\n")
file.close() 

##########################################################################################################################################################