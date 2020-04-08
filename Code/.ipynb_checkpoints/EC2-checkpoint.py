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
rep_times = n_cores
# rep_times = 48
# rep_times = 32
# rep_times = 16
# rep_times = 96

# region_parallel = True
region_parallel = False
##########################################################################################################################################################


# sd_u_O = 0.3
sd_u_O = 30

# thre_range = [95, 100, 105, 110]
thre_range = [80, 90, 100, 110]
# thre_range = [75, 80, 85, 90, 100, 105, 110, 120]
# thre_range = [80, 85, 90, 95, 100]
# thre_range = [100, 105, 110, 115]
    


# sd_R_range = [0, 10, 20]
sd_R_range = [10]
# sd_R_range = [0, 5]
# sd_R_range = [10, 15]
# sd_R_range = [20, 25]
# sd_R_range = [25, 50, 100]

# day_range = [3, 7, 14] # 14
# day_range = [6, 10, 14]
day_range = [4, 8, 12]


with_MF = False
with_NO_MARL = True

##########################################################################################################################################################
pattern_seed = 2
w_O = 0.5
w_A = 1
##########################################################################################################################################################
#           M_in_R, MR, poisO, u_O_u_D
DGP_choice = [True, False, True, 10]
sd_O = None
sd_D = None
l = 5
T = None
# t_func = t_func_peri
t_func = None
##########################################################################################################################################################
# NN
n_layer = 3
max_iteration = 1001
Learning_rate = 5e-4
w_hidden = 30
batch_size = 32
##########################################################################################################################################################
printR("final sd_R trend for" + str(sd_R_range) + " the same \n")

shared_setting = "Basic setting:" + "[T, rep_times, sd_O, sd_D, sd_u_O, w_O, w_A, [M_in_R, mean_reversion, poisO, u_O_u_D], sd_R_range, t_func] = " + str([T, rep_times, sd_O, sd_D,  sd_u_O, w_O, w_A, DGP_choice, sd_R_range, t_func]) + "\n"
printR(shared_setting)

##########################################################################################################################################################


results = []
res_real = []
# for pattern_seed in [4, 5, 6, 7, 8, 9, 10]:
for day in day_range:
    for sd_R in sd_R_range:
        T = day * 48
        res = []
        print_setting = DASH + "[pattern_seed, day, sd_R] = " + str([pattern_seed, day, sd_R]) + "\n"
        printR(print_setting)
        N_targets = simu(pattern_seed = pattern_seed, l = l, T = T, t_func = t_func, # Setting - general
                 thre_range = thre_range, DGP_choice = DGP_choice, # Setting - general
                 sd_D = sd_D, sd_R = sd_R, sd_O = sd_O, sd_u_O = sd_u_O, # Setting - noise
                  w_A = w_A, w_O = w_O,  # Setting - spatial
                  # fixed
                  n_cores = n_cores, OPE_rep_times = rep_times, inner_parallel = region_parallel, # Parallel
                  CV_QV = True, penalty = [[1e-4, 5e-5], [1e-4, 5e-5]],  # 3 * 3 * (n_cv + 1)
#                  CV_QV = True, penalty = [[1e-3, 1e-4, 3e-4, 6e-4], [1e-3, 1e-4, 3e-4, 6e-4]],
#                       CV_QV = False, penalty = [[1e-4], [1e-4]], 
#                       penalty_NMF = [[1e-3, 1e-4, 1e-5], [1e-3, 1e-4, 1e-5]], 
                  penalty_NMF = [[1e-3], [1e-3]], # Q-V hyperparameters
                  w_hidden = w_hidden, n_layer = n_layer,  # NN hyperparameters
                  batch_size = batch_size, max_iteration = max_iteration, Learning_rate = Learning_rate,  # NN training
                  dim_S_plus_Ts = 3 + 3, epsilon = 1e-6, # Fixed
                  with_MF = with_MF, with_NO_MARL = with_NO_MARL
                  )
        # r: a list (len-N_target) of list of [bias, std, MSE] (each of the three is a vector)
        res.append(N_targets)
        res_real.append(arr([a[2] for a in N_targets])) # N_targets * N_method for MSE
        for i in range(len(res_real)):
            print(res_real[i])
            print("\n")
        print("time spent until now:", np.round((now() - a)/60, 1), "mins", "\n")
        printR(str(EST()))
        results.append(res)
    
with open("0406_sd_R.txt", "wb") as fp:
    pickle.dump(results, fp)
    
# with open("0330.txt", "rb") as fp:
#     b = pickle.load(fp)

##########################################################################################################################################################

