library(matrixStats)
norm_vec <- function(x) sqrt(sum(x^2))

## Helper Functions  ####################################
policy <-function(XI, vlam){
  J = length(vlam)
  level = 0
  for (lam in vlam) {
    if (XI < lam) {
      return(level + 1)
    }else{
      level = level + 1
    }
  }
  return(J + 1)
}


projectEk <- function(lam, xi_k, lam_M){
  # The project operator in Constrained SPSA
 # Now only support J  = 3 for our experiments;
  lam2 = lam[1]
  lam3 = lam[2]
  J = 3
  gap = xi_k * sqrt(J - 1)
 if (lam2 < gap){
   if (lam3 < 2 * gap) {
     return(c(gap, 2 * gap))
   }else if(lam3 > lam_M - gap){
     return(c(gap, lam_M - gap))
   }else{
     return(c(gap,lam3))
   }
 }
  if(lam2 > gap){
    if( lam2 < lam_M - 2 * gap){
      if(lam3 > lam_M - gap){
        return(c(lam2, lam_M - gap))
      }else if(lam3 > lam2 + gap){
        return(c(lam2, lam3))
      }else if(lam2 + lam3 < 3 * gap){
        return(c(gap, 2 * gap))
      }
    }
  }
  if(lam2 + lam3 < lam_M * 2 - 3 * gap){
    # project onto that line
    # lam22 + lam33 = lam2 + lam3
    # lam33 - lam22 = gap
    return(c((lam2 + lam3 - gap) / 2, (lam2 + lam3 + gap) / 2 ))
  }else{
    return(c(lam_M - 2 * gap, lam_M - gap))
  }
}

##  Trasition Models ####################################


freq_traj <- function(X_lt, As, theta, f1 = NA, X_l = NA){
  # Given the current state and following actions, generate a trajector (frequentist)
  T_fwd = length(As)
  states = matrix(rep(0, 3 * T_fwd), nrow = 3, ncol = T_fwd)
  
  for (t in 1:T_fwd) {
    A = As[t]
    X_lt_next = f(X_lt, A + 1, theta)
    states[1:3, t] = X_lt_next
    X_lt = X_lt_next
  }
  return(states)
}



freq_band <-function(X_lt, As, theta, f1 = NA, X_l = NA, rep = 1000){
  # Given the current state and following actions, generate the PIs (frequentist)
  T_fwd = length(As)
  X_I  = matrix(rep(0, rep * T_fwd), nrow = rep, ncol = T_fwd)
  
  for (i in 1:rep) {
    XIi = freq_traj(X_lt, As, theta, f1 = NA, X_l = NA)[2,]
    X_I[i,] =  XIi
  }
  return(colQuantiles(X_I, probs = c(0.025, 0.975), na.rm = T))
}

# 
# Code for testing the PI
l = 1
Xl = data[[l]]
tl = Xl$tl[1]

I_confirmed = Xl$infected
I_confirmed = c(rep(0, lag), I_confirmed)
# define the state vairables, based on the discussion in Sec 2.3
XI = (shift(I_confirmed, n = lag, fill = NA, type="lead") - I_confirmed)
XR = I_confirmed
XS = 1 - XI - XR
A = c(rep(0, lag), Xl$action)

X = cbind(XS,XI,XR)

X_lt = X[30,1:3]
As = A[30:36]
# freq_traj(X_lt, As, theta, f1 = NA, X_l = NA)

# theta = params_shift[1:8, 1]
freq_band(X_lt, As, theta, f1 = NA, X_l = NA, rep = 1000) * 1e12
cat(X[30:36,2] * 1e12)





#### Main Algorithms ##################################

SPSA <- function(theta, S, w, tol, zeta, xi, tao, t0, vlam,  lam_M, T_end = 180, J = 3){
  # Algorithm 1: solve the optimal policy with planning
  # Q: ifferent city will have different vlam_1? no?
  # Args:
  #   S: (X_lt, X_l), where X_lt = (X^S, X^I, X^R), and X_l = GDP (for now)
  # # theta: (gamma, sigma_R, beta, sigma_S)
  # Returns:

  ######################################
  ######################################
  # Initialization
  set.seed(1)
  k = 1
  delta = 1e10
  
  gamma = theta[1]
  sigma_R = theta[2]
  beta  = theta[3 + A - 1]
  sigma_S = theta[3 + J + A - 1]
  
  X_l = S[4:length(S)]
  while (1) {
    Delta = rbinom(J, 1, .5)
    zeta_k = zeta / k^tau
    xi_k = xi / k
    vlam_1 = vlam + Delta * xi_k
    vlam_2 = vlam - Delta * xi_k
    V1_k = V2_k = 0
    
    for (m in M) {
      V1_kk = V2_kk = 0
      X_lt1 = X_lt2 = S[1:3]
      for (tt in t0:T_end) {
        A1 =  policy(X_lt1, vlam_1)
        X_lt1_old = X_lt1
        X_lt1 = f(X_lt1, A1, theta) # only calculate dynamic state variables
        Y1 = g(X_lt1_old, A1, X_lt1, X_l, w)
        V1_kk  = V1_kk + Y1
        
        A2 = policy(X_lt2, vlam_2)
        X_lt2_old = X_lt2
        X_lt2 = f(X_lt2, A2, theta)
        Y2 = g(X_lt2_old, A2, X_lt2, X_l, w)
        V2_kk  = V2_kk + Y2
      }
      V1_k = V1_k + V1_kk / (T_end - t0 + 1)
      V2_k = V2_k + V2_kk / (T_end - t0 + 1)
    }
    V1_k = V1_k / M
    V2_k = V2_k / M
    
    vlam_old = vlam
    vlam = vlam + (V1_k - V2_k) / (2 * xi_k) * zeta_k * Delta
    vlam = projectEk(vlam, xi_k, lam_M)
    k = k + 1
    delta = norm_vec(vlam - vlam_old) /  norm_vec(vlam_old)
  }
  
  A = policy(X_l, vlam)
  
  return(c(vlam, A))
}


Pareto <-function(w_list, alpha, theta, S, w, tol, zeta, xi, tao, t0, T_end, J, lam_M){
  # Computes the sample covariance between two vectors.
  #
  # Args:
  #   w_list: a list of weight
  #   alpha: significance level of the predictive band
  #
  # Returns:
  #   a list of {w, lam, mean of R and C, mean trajectory of R and C, predictive band of R and C}
  r = list()
  for (w in w_list) {
    vlam = SPSA(theta, S, w, tol, zeta, xi, tao, t0, T_end, J, lam_M)
    r_w = list()
    r_w["w"] = w
    r_w["lam"] = vlam
    R_mean = rep(0, T_end - t0 + 1)
    R_upper = rep(0, T_end - t0 + 1)
    R_lower = rep(0, T_end - t0 + 1)
    C_mean = rep(0, T_end - t0 + 1)
    C_upper = rep(0, T_end - t0 + 1)
    C_lower = rep(0, T_end - t0 + 1)
    for (tt in t0: T_end){
      R_tt = rep(0, M)
      C_tt = rep(0, M)
      for (m in 1:M) {
        A =  policy(XI, vlam)
        XI_old = XI
        XI = f(XI, A, theta)
        R_tt[m] = g_R(XI1_old, XI1)
        C_tt[m] = g_C(A, X_l)
      }
      R_mean[tt - t0 + 1] = mean(R_tt)
      R_lower[tt - t0 + 1] = quantile(R_tt,alpha / 2)
      R_upper[tt - t0 + 1] = quantile(R_tt,1 - alpha / 2)
      C_mean[tt - t0 + 1] = mean(C_tt)
      C_lower[tt - t0 + 1] = quantile(C_tt,alpha / 2)
      C_upper[tt - t0 + 1] = quantile(C_tt,1 - alpha / 2)
    }
    r_w["R_mean_band"] = R_mean
    r_w["R_upper"] = R_upper
    r_w["R_lower"] = R_lower
    r_w["R_mean"] = mean(R_mean)
    
    r_w["C_mean_band"] = C_mean
    r_w["C_upper"] = C_upper
    r_w["C_lower"] = C_lower
    r_w["C_mean"] = mean(C_mean)
    r[w] = r_w
  }
  return(r)
}
