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

rep_times = 3

l = 5
T = None

sd_O = 10 #1
sd_D = 10 #1
sd_R = 5 #1
sd_u_O = 0.2 #0.4

w_O = .5
w_A = 1


n_layer = 3
max_iteration = 1501
Learning_rate = 1e-4
# n_layer = 3
# max_iteration = 2001
# Learning_rate = 1e-4
lam = 1e-4

thre_range = [80,  90,  100, 110, 120, 130]
day_range = [3, 7, 14]
l_range = [3, 4, 5, 6, 7]
inner_parallel = True
path = "0329.txt"
##########################################################################################################################################################
shared_setting = "Basic setting:" + "[T, sd_O, sd_D, sd_R, sd_u_O, w_O, w_A, lam, simple, M_in_R, u_O_u_D, mean_reversion, day_range, thre_range] = " + str([T, sd_O, sd_D, sd_R, sd_u_O, w_O, w_A, lam, simple, M_in_R, u_O_u_D, mean_reversion, day_range, thre_range]) + "\n"
print(shared_setting)


##########################################################################################################################################################
settings = []
for pattern_seed in range(2):
    settings.append([pattern_seed, sd_R])

##########################################################################################################################################################

    
results = []    
for setting in settings:
    res = []
    for l in l_range:
        T = 14 * 48 #int(day * 48)
        pattern_seed, sd_R = setting
        print_setting = DASH + "[pattern_seed, T, sd_R] = " + str([pattern_seed, T, sd_R]) + "\n"
        print(print_setting)
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
                  print_flag_target = False
                  )
        # a list (len-N_target) of list of [bias, std, MSE] (each is a vector). 
        print("time spent until now:", np.round((now() - a)/60, 1), "mins", "\n")
        res.append(r)
    results.append(res)
    
with open("0330.txt", "wb") as fp:
    pickle.dump(results, fp)
    
# with open("0330.txt", "rb") as fp:
#     b = pickle.load(fp)
# How to interpret?

##########################################################################################################################################################

