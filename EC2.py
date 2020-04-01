##########################################################################################################################################################
from _utility import *
from _uti_basic import *
from weight import *
from simu_funs import *
from main import *
a = now()
# export openblas_num_threads=1; export OMP_NUM_THREADS=1; python EC2.py
printR(str(EST()) + "; num of cores:" + str(n_cores) + "\n")

##########################################################################################################################################################
# rep_times = n_cores

rep_times = 1

l = 5
T = None

sd_O = 10 #1
sd_D = 10 #1
sd_R = 5 #1
sd_u_O = 0.2 #0.4

w_O = 1
w_A = 1


n_layer = 3
max_iteration = 1001
Learning_rate = 1e-4
# n_layer = 3
# max_iteration = 2001
# Learning_rate = 1e-4
lam = 1e-4

thre_range = [80]
day_range = [3, 7, 14]
inner_parallel = True
path = "0329.txt"
##########################################################################################################################################################
shared_setting = "Basic setting:" + "[T, sd_O, sd_D, sd_R, sd_u_O, w_O, w_A, lam, simple, M_in_R, u_O_u_D, mean_reversion, day_range, thre_range, poisO] = " + str([T, sd_O, sd_D, sd_R, sd_u_O, w_O, w_A, lam, simple, M_in_R, u_O_u_D, mean_reversion, day_range, thre_range, poisO]) + "\n"
print(shared_setting)


##########################################################################################################################################################
settings = []
for pattern_seed in range(2):
    for sd_OD in [.5, 5, 10, 20]:
        settings.append([pattern_seed, sd_OD])

##########################################################################################################################################################

    
results = []
res_real = []
for setting in settings:
    res = []
    T = 48 * 4
#     for day in day_range:
#         T = int(day * 48)
    pattern_seed, sd_OD = setting
    sd_O = sd_D = sd_OD
    print_setting = DASH + "[pattern_seed, sd_OD] = " + str([pattern_seed, sd_OD]) + "\n"
    print(print_setting)
    N_targets = simu(pattern_seed = pattern_seed, l = l, T = T, time_dependent = False, # Setting - general
             thre_range = thre_range, # Setting - general
             sd_D = sd_D, sd_R = sd_R, sd_O = sd_O, sd_u_O = sd_u_O, # Setting - noise
              w_A = w_A, w_O = w_O,  # Setting - spatial
              # fixed
              n_cores = 1, OPE_rep_times = 1, inner_parallel = False, # Parallel
             CV_QV = True, penalty = [[0.01, 0.1], [lam]], penalty_NMF = [[1e-3], [1e-3]], # Q-V hyperparameters
              w_hidden = 30, n_layer = n_layer,  # NN hyperparameters
              batch_size = 32, max_iteration = max_iteration, Learning_rate = Learning_rate,  # NN training
              dim_S_plus_Ts = 3 + 3, epsilon = 1e-6, # Fixed
              print_flag_target = False
              )
    # r: a list (len-N_target) of list of [bias, std, MSE] (each of the three is a vector)
    res.append(N_targets)
    res_real.append(arr([a[2] for a in N_targets]))
    print(res_real)
    print("time spent until now:", np.round((now() - a)/60, 1), "mins", "\n")
    results.append(res)
    
with open("0330.txt", "wb") as fp:
    pickle.dump(results, fp)
    
# with open("0330.txt", "rb") as fp:
#     b = pickle.load(fp)
# How to interpret?

##########################################################################################################################################################

