##########################################################################################################################################################
from _utility import *
from _uti_basic import *
from weight import *
from simu_funs import *
from main import *
a = now()
# export openblas_num_threads=1; export OMP_NUM_THREADS=1
printR(str(EST()) + "; num of cores:" + str(n_cores) + "\n")

##########################################################################################################################################################
# rep_times = n_cores
rep_times = 2

l = 5
lam = 1e-4

sd_R = None
sd_O = 2
sd_D = 2
sd_u_O = 0.4

w_O = 1
w_A = 1

inner_parallel = True
path = "0329.txt"
##########################################################################################################################################################
file = open(path, 'w') 
shared_setting = "Basic setting:" + "[sd_O, sd_D, sd_R, sd_u_O, w_O, w_A, lam] = " + str([sd_O, sd_D, sd_R, sd_u_O, w_O, w_A, lam]) + "\n"
print(shared_setting)
print(shared_setting, file = file)

##########################################################################################################################################################
#             for T in [int(14 * 48)]: # , int(14 * 48 * 2) # int(14 * 48 / 2), 
#     settings.append([None, sd_D, w_A, T])
settings = []
for pattern_seed in range(10):
    for T in [14 * 48]:#, 14 * 48 * 2, 7 * 48, 
        for sd_R in [0, 2]:
                    settings.append([pattern_seed, T, sd_R])

##########################################################################################################################################################
for setting in settings:
    pattern_seed, T, sd_R = setting
    print_setting = DASH + "[pattern_seed, T, sd_R] = " + str([pattern_seed, T, sd_R]) + "\n"
    print(print_setting)
    print(print_setting, file = file)
    r = simu(pattern_seed = pattern_seed, l = l, T = T, time_dependent = False, # Setting - general
             thre_range = [12, 9, 15], # Setting - general
             sd_D = sd_D, sd_R = sd_R, sd_O = sd_O, sd_u_O = sd_u_O, # Setting - noise
              w_A = w_A, w_O = w_O,  # Setting - spatial
              # fixed
              n_cores = n_cores, OPE_rep_times = rep_times, inner_parallel = inner_parallel, # Parallel
              penalty = [[lam], [lam]], penalty_NMF = [[1e-3], [1e-3]], # Q-V hyperparameters
              w_hidden = 30, n_layer = 2,  # NN hyperparameters
              batch_size = 32, max_iteration = 1001, Learning_rate = 1e-3,  # NN training
              dim_S_plus_Ts = 3 + 3, epsilon = 1e-6, # Fixed
              file = file, print_flag_target = False
              )
    print("time spent until now:", np.round((now() - a)/60, 1), "mins", "\n")
file.close() 

##########################################################################################################################################################

