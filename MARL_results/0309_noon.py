# from _uti_basic import *
from _utility import *
from weight import *
from simu_funs import *
from main import *
os.environ["OMP_NUM_THREADS"] = "1"

l = 5
T = 48 * 14
rep_times = n_cores

a = now()
lam = 0.01
w_hidden = 30

for w_spatial in [0.2, 1]:
	print(DASH, "[lam, w_hidden] = ", [lam, w_hidden])
	r = simu(pattern_seed = 0, l = l, T = T, sd_D = 0, sd_R = 0, # Setting - general
	         simple_grid_neigh = False, w_A = w_spatial, w_O = w_spatial,  # Setting - spatial
	          n_cores = n_cores, OPE_rep_times = rep_times, inner_parallel = False, # Parallel
	          penalty = [[lam], [lam]], w_hidden = w_hidden, n_layer = 2,  # NN - hyperparameters
	          batch_size = 32, max_iteration = 1001, Learning_rate = 1e-3, test_num = 0,  # NN training
	          dim_S_plus_Ts = 3 + 3, epsilon = 1e-6
	          )
	print( "time spent until now:", np.round((now() - a)/60, 1), "mins", "\n")


--------------------------------------
 [lam, w_hidden] =  [0.01, 30]
15:09, 03/09; num of cores:16
1 1 1 1 0

1 0 1 1 0

1 1 1 1 0

0 0 1 1 1

1 1 0 1 0

MC-based mean [average reward] and its std: [6.384e+00 1.000e-03]
DR, IS, Susan, DR_NS
 bias: [0.188 0.183 0.252 0.186]
 std: [0.036 0.033 0.02  0.035]
 MSE: [0.191 0.186 0.253 0.189]
time spent until now: 51.9 mins


--------------------------------------
 [lam, w_hidden] =  [0.01, 30]
16:01, 03/09; num of cores:16
1 1 1 1 0

1 0 1 1 0

1 1 1 1 0

0 0 1 1 1

1 1 0 1 0

^[[A^[[A^[[A^[[A^[[A^[[A^[[AMC-based mean [average reward] and its std: [6.818e+00 1.000e-03]
DR, IS, Susan, DR_NS 
 bias: [0.211 0.193 0.046 0.202] 
 std: [0.03  0.027 0.02  0.029] 
 MSE: [0.213 0.195 0.05  0.204]
time spent until now: 102.9 mins 
