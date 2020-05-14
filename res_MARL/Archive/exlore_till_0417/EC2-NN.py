##########################################################################################################################################################
from _utility import * 
from _uti_basic import * 
from weight import * 
from simu_funs import *  
from main import *
a = now()
#export openblas_num_threads=1; export OMP_NUM_THREADS=1; python EC2.py
printR(str(EST()) + "; num of cores:" + str(n_cores) + "\n")

##########################################################################################################################################################
rep_times = n_cores
printR("NN tuning!")

# rep_times = 96
region_parallel = False
# parallel = regions[25], reps[more. 96], settings[pattern_seed, day_range]
# CV_QV, penalty_NMF, w_hidden(paras for NN)

l = 5
T = None

sd_OD = 10
sd_D = sd_O = sd_OD

sd_R = None #
sd_u_O = 0.3 #0.4

w_O = .5
w_A = 1

#           M_in_R, MR, poisO, simple, u_O_u_D
DGP_choice = [True, False, True, False, 10]


# thre_range = [80, 90, 100, 110]
thre_range = [80, 90, 100, 110]
# day_range = [5, 12] # 7, 10, 
day_range = [7, 14]

sd_R = 10

##########################################################################################################################################################
# NN
n_layer = 3
max_iteration = 1501
Learning_rate = 1e-4
w_hidden = 30
batch_size = 32

##########################################################################################################################################################
shared_setting = "Basic setting:" + "[T, rep_times, sd_O, sd_D, sd_R, sd_u_O, w_O, w_A, [M_in_R, mean_reversion, poisO, simple, u_O_u_D]] = " + str([T, rep_times, sd_O, sd_D, sd_R, sd_u_O, w_O, w_A, DGP_choice]) + "\n"
printR(shared_setting)

##########################################################################################################################################################


results = []
res_real = []
pattern_seed = 2
for day in day_range:
    for n_layer in [2, 3]:
        for w_hidden in [30, 50]:
            for NN_setting in [[1001, 1e-3], [2001, 1e-4]]:
                max_iteration, Learning_rate = NN_setting
                T = day * 48
                res = []
                print_setting = DASH + "[pattern_seed, day, sd_R, n_layer, w_hidden, NN_setting] = " + \
                    str([pattern_seed, day, sd_R, n_layer, w_hidden, NN_setting]) + "\n"
                printR(print_setting)
                N_targets = simu(pattern_seed = pattern_seed, l = l, T = T, time_dependent = False, # Setting - general
                         thre_range = thre_range, DGP_choice = DGP_choice, # Setting - general
                         sd_D = sd_D, sd_R = sd_R, sd_O = sd_O, sd_u_O = sd_u_O, # Setting - noise
                          w_A = w_A, w_O = w_O,  # Setting - spatial
                          # fixed
                          n_cores = n_cores, OPE_rep_times = rep_times, inner_parallel = region_parallel, # Parallel
                          CV_QV = True, penalty = [[1e-4, 5e-5], [1e-4, 5e-5]],  # 3 * 3 * (n_cv + 1)
    #                       CV_QV = False, penalty = [[1e-4], [1e-4]], 
                          penalty_NMF = [[1e-3, 1e-4, 1e-5], [1e-3, 1e-4, 1e-5]], 
    #                       penalty_NMF = [[1e-3], [1e-3]], # Q-V hyperparameters
                          w_hidden = w_hidden, n_layer = n_layer,  # NN hyperparameters
                          batch_size = batch_size, max_iteration = max_iteration, Learning_rate = Learning_rate,  # NN training
                          dim_S_plus_Ts = 3 + 3, epsilon = 1e-6, # Fixed
                          print_flag_target = False
                          )
                # r: a list (len-N_target) of list of [bias, std, MSE] (each of the three is a vector)
                res.append(N_targets)
                res_real.append(arr([a[2] for a in N_targets])) # N_targets * N_method for MSE
                for i in range(len(res_real)):
                    print(res_real[i])
                    print("\n")
                print("time spent until now:", np.round((now() - a)/60, 1), "mins", "\n")
                results.append(res)
    
with open("0402.txt", "wb") as fp:
    pickle.dump(results, fp)
    
# with open("0330.txt", "rb") as fp:
#     b = pickle.load(fp)

##########################################################################################################################################################

