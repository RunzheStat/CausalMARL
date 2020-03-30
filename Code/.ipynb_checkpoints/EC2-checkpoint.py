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
T = 14 * 48

sd_O = 5 #1
sd_D = 10 #1
sd_R = 10 #1
sd_u_O = 0.2 #0.4

w_O = 1
w_A = 1


n_layer = 2
max_iteration = 1001
Learning_rate = 1e-3
# n_layer = 3
# max_iteration = 2001
# Learning_rate = 1e-4
lam = 1e-5

thre_range = [100, 80, 85, 90, 95, 105, 110]
# thre_range = [100, -4, -3, -2, -1, 90]

inner_parallel = True
path = "0329.txt"
##########################################################################################################################################################
file = open(path, 'w') 
shared_setting = "Basic setting:" + "[T, sd_O, sd_D, sd_R, sd_u_O, w_O, w_A, lam, simple, M_in_R, u_O_u_D, mean_reversion] = " + str([T, sd_O, sd_D, sd_R, sd_u_O, w_O, w_A, lam, simple, M_in_R, u_D_u_O, mean_reversion]) + "\n"
print(shared_setting)
print(shared_setting, file = file)

##########################################################################################################################################################
#             for T in [int(14 * 48)]: # , int(14 * 48 * 2) # int(14 * 48 / 2), 
#     settings.append([None, sd_D, w_A, T])
settings = []
for pattern_seed in range(5):
    for T in [14 * 48]:#, 14 * 48 * 2, 7 * 48, 
        settings.append([pattern_seed, T, sd_R])

##########################################################################################################################################################
for setting in settings:
    pattern_seed, T, sd_R = setting
    print_setting = DASH + "[pattern_seed, T, sd_R] = " + str([pattern_seed, T, sd_R]) + "\n"
    print(print_setting)
    print(print_setting, file = file)
    r = simu(pattern_seed = pattern_seed, l = l, T = T, time_dependent = False, # Setting - general
             thre_range = thre_range, # Setting - general
             sd_D = sd_D, sd_R = sd_R, sd_O = sd_O, sd_u_O = sd_u_O, # Setting - noise
              w_A = w_A, w_O = w_O,  # Setting - spatial
              # fixed
              n_cores = n_cores, OPE_rep_times = rep_times, inner_parallel = inner_parallel, # Parallel
              penalty = [[lam], [lam]], penalty_NMF = [[1e-3], [1e-3]], # Q-V hyperparameters
              w_hidden = 30, n_layer = n_layer,  # NN hyperparameters
              batch_size = 32, max_iteration = max_iteration, Learning_rate = Learning_rate,  # NN training
              dim_S_plus_Ts = 3 + 3, epsilon = 1e-6, # Fixed
              file = file, print_flag_target = False
              )
    print("time spent until now:", np.round((now() - a)/60, 1), "mins", "\n")
file.close() 

##########################################################################################################################################################

