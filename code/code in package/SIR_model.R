
#' f1_func
#' Heterogeneity function control transmission rate \beta
#'
#' @param data data as a list with L data frames for each city
#' @param location integer
f1_func <- function(data=data, location=1){

  return(1)
}

#' f2_func
#' Heterogeneity function control recover rate \gamma
#'
#' @param data data as a list with L data frames for each city
#' @param location integer
f2_func <- function(data=data, location=1){

  return(1)
}


#' getSIRMAP
#' Use observations until A_{t0-1} and X_{t0} to train the parameter theta.
#'
#' @param data data as a list with L data frames for each city, data$infected, $suspected, $removed, $action, $population
#' @param t0 Use data until A_{t0-1} and S_{t0} to train the parameters
#' @param lag shifted time between data and self defined state variable
#' @param J integer, number of action levels
#' @param u_2 = 0.5 prior for relative reansmission rate of action 1 to action 0
#' @param u_3 = 0.1 prior for relative reansmission rate of action 2 to action 0
#' @param region_index vector of integers, indicating which region you are gonna select for estimation
#' @param MA = 0.4 central weight for smooth moving average method
#' @param echo boolean variable indicating if the zlts/zltr should be output during the process
#' @param t_start = 0 vector of integers
getSIRMAP <- function(data=data_cn, t0=25, lag=8, J=3, 
                      u_2 = 0.5, u_3 = 0.1, region_index=c(1:50), 
                      MA_index = NA, echo = F, t_start = 0,
                      poisson = T){
  ## Use data until A_{t0-1} and S_{t0}
  # t0: t0 + lag should be less than max length (42 for now)
  t_max = dim(data[[1]])[1]
  if((t0) > t_max){
    stop(paste("The input t0 should be an integer from 1+tl+t_start to", t_max,".\n"))
  }
  # Parameter Initialization prior
  aR = 3
  bR = 2 * 1e-10
  aS = rep(3, J)
  bS = rep(2, J) * 1e-10
  mu.gamma = .0117
  sigma.gamma = .1
  mu.beta = c(3.15, 3.15 * u_2, 3.15 * u_3) * mu.gamma
  sigma.beta = rep(0.5,J) # 0.33 based on Peter_Song
  # load function f(X_l)
  f1_func(data,location=1) # range(1,50)
  f2_func(data,location=5)
  
  if(poisson){
    # empty vector initialization
    # f1 <- rep(1,N)
    # f2 <- rep(1,N)
    gamma_data = list()
    gamma.count = 1
    beta_data = list()
    beta.count = list()
    for (i in 1:J) {
      beta_data[[i]] = list()
      beta.count[[i]] = 1
    }
    
    for (l in region_index){
      f1 = f1_func(data, l)
      f2 = f2_func(data, l)
      Xl = addTltoData(data[[l]])
      tl = Xl$tl[1]
      popu = Xl$population[1]
      
      # If there's no available data, then skip this city
      if(is.na(tl)){
        next
      }
      
      # 9:50 shift to 1:42
      Xl_shifted = shift_state(Xl, lag = lag, MA = MA_index, popu = popu)
      XS = Xl_shifted$XS 
      XI = as.integer(Xl_shifted$XI )
      XR = as.integer(Xl_shifted$XR )
      A =  Xl_shifted$A
      
      for (t in (tl + t_start):(t0-1)){
        
        Xti = XI[t] 
        Xtr = XR[t] 
        Xts = XS[t] 
        
        Xti_p1 = XI[t+1] 
        Xtr_p1 = XR[t+1] 
        Xts_p1 = XS[t+1] 
        
        if ( Xti > 1e-11 ){#  & Zltr > 0
          gamma_data[[gamma.count]] = c(f2 * Xti, max(Xtr_p1 - Xtr, 0) )
          gamma.count = gamma.count + 1
          i = A[t]
          beta_data[[i]][[beta.count[[i]]]] = c(f1 * Xts * Xti / popu,  max(Xts - Xts_p1, 0))
          beta.count[[i]] = beta.count[[i]] + 1
          
          if(echo)cat("l:",l,"t",t,"gamma",c(f2 * Xti, Xtr_p1 - Xtr),"|| beta:",  c(f1 * Xts * Xti, Xts - Xts_p1), "action=", i,"\n")
        }
      }
    }
    
    # gamma
    gamma_data = do.call(rbind, gamma_data)
    gamma_data = as.data.frame(gamma_data)
    names(gamma_data)  = c("x", "y")
    gamma_MAP = MAP(gamma_data, prior.mu = mu.gamma, prior.sigma = sigma.gamma, low = .005, high = 0.3, is_gamma = T)$maximum
    beta_MAP = rep(0,3)
    beta_data_new = list()
    for (i in 1:J) {
      beta_data_new[[i]] = do.call(rbind, beta_data[[i]])
      beta_data_new[[i]] = as.data.frame(beta_data_new[[i]])
      names(beta_data_new[[i]])  = c("x", "y")
      beta_MAP[i] = MAP(data = beta_data_new[[i]], prior.mu = mu.beta[i], 
                        prior.sigma = sigma.beta[i], low = 0, high = 1, is_gamma = F)$maximum
    }
    R0 = beta_MAP / gamma_MAP
    parameter = c(gamma_MAP, beta_MAP, R0)
    return(round(parameter, 6))
    
  }else{ #normal
    # empty vector initialization
    # f1 <- rep(1,N)
    # f2 <- rep(1,N)
    MA <- c(0)
    MB <- c(0)
    MC <- c(0)
    MAI <- rep(0,J)
    MBI <- rep(0,J)
    MCI <- rep(0,J)
    
    
    
    # shifted version
    for (l in region_index){
      f1 = f1_func(data, l)
      f2 = f2_func(data, l)
      Xl = addTltoData(data[[l]])
      tl = Xl$tl[1]
      
      # If there's no available data, then skip this city
      if(is.na(tl)){
        next
      }
      
      # 9:50 shift to 1:42
      Xl_shifted = shift_state(Xl, lag = lag, MA = MA_index)
      XS = Xl_shifted$XS
      XI = Xl_shifted$XI
      XR = Xl_shifted$XR
      A =  Xl_shifted$A
      
      for (t in (tl + t_start):(t0-1)){
        
        Xti = XI[t]
        Xtr = XR[t]
        Xts = XS[t]
        
        Xti_p1 = XI[t+1]
        Xtr_p1 = XR[t+1]
        Xts_p1 = XS[t+1]
        
        if ( Xti > 1e-11 ){#  & Zltr > 0
          Zltr =  (Xtr_p1 - Xtr)/(f2*Xti)
          Zlts =  -(Xts_p1 - Xts)/(f1*Xti*Xts)
          
          MA = MA + 1 # counter
          MB = MB + Zltr
          MC = MC + Zltr^2
          i = A[t]
          
          MAI[i] = MAI[i] + 1 # counter
          MBI[i] = MBI[i] + Zlts
          MCI[i] = MCI[i] + Zlts^2
          if(echo)cat("l:",l,"t",t,"xti",Xti,": Zltr=",Zltr,"  .","Zlts=", Zlts, "action=", i,"\n")
          
        }
      }
    }
    
    
    Gt0 = MA
    GtI = MAI
    cat("GTI",sum(GtI))
    # Posterior mean
    MA = MA + 1/(sigma.gamma^2) # counter
    MB = MB + mu.gamma/sigma.gamma
    MC = MC + (mu.gamma/sigma.gamma)^2
    MAI = MAI + 1/(sigma.beta^2)
    MBI = MBI + mu.beta/sigma.beta
    MCI = MCI + (mu.beta/sigma.beta)^2
    
    # gamma
    gamma.mode = MB / MA
    gamma.v = (2*aR + Gt0)
    gamma.mu = MB / MA
    gamma.sigma2 = (2*MA*(bR+MC)-MB^2)/(MA^2*(gamma.v))
    gamma.var = (gamma.v / (gamma.v-2) )* gamma.sigma2
    gamma.std = sqrt(gamma.var)
    gamma.95cilb = qt(c(.025), df=gamma.v) * sqrt(gamma.sigma2) + gamma.mu
    gamma.95ciub = qt(c(.975), df=gamma.v) * sqrt(gamma.sigma2) + gamma.mu
    gamma = cbind(mode = gamma.mode, std = gamma.std, CILB = gamma.95cilb, CIUB = gamma.95ciub)
    row.names(gamma) = "gamma"
    
    # beta
    beta.mode = MBI / MAI
    beta.v = (2*aS + GtI)
    beta.mu = MBI / MAI
    beta.sigma2 = (2*MAI*(bS+MCI)-MBI^2)/(MAI^2*(beta.v))
    beta.var = (beta.v) / (beta.v-2) * beta.sigma2
    beta.std = sqrt(beta.var)
    beta.95cilb = qt(c(.025), df=beta.v) * sqrt(beta.sigma2) + beta.mu
    beta.95ciub = qt(c(.975), df=beta.v) * sqrt(beta.sigma2) + beta.mu
    beta = cbind(mode = beta.mode, std = beta.std, CILB = beta.95cilb, CIUB = beta.95ciub)
    row.names(beta)=c("beta1","beta2","beta3")
    
    # sigma2.gamma
    sigma2.gamma.a = aR + Gt0/2
    sigma2.gamma.b = bR + MC/2-MB^2/(2*MA)
    sigma2.gamma.mode = sigma2.gamma.b/(sigma2.gamma.a + 1)
    sigma2.gamma.mu = sigma2.gamma.b/(sigma2.gamma.a - 1)
    sigma2.gamma.var = sigma2.gamma.b^2/( (sigma2.gamma.a - 1)^2 * (sigma2.gamma.a - 2))
    sigma2.gamma.std = sqrt(sigma2.gamma.var)
    sigma2.gamma.95cilb = qinvgamma(c( .025), shape=sigma2.gamma.a, rate =sigma2.gamma.b)
    sigma2.gamma.95ciub = qinvgamma(c( .975), shape=sigma2.gamma.a, rate =sigma2.gamma.b)
    sigma2.gamma = cbind(mode = sigma2.gamma.mode, std = sigma2.gamma.std,
                         CILB = sigma2.gamma.95cilb, CIUB = sigma2.gamma.95ciub)
    row.names(sigma2.gamma) = "sigma2.gamma"
    
    
    
    # sigma2.beta
    sigma2.beta.a = aS + GtI/2
    sigma2.beta.b = bS + MCI/2-MBI^2/(2*MAI)
    sigma2.beta.mode = sigma2.beta.b/(sigma2.beta.a + 1)
    sigma2.beta.mu = sigma2.beta.b/(sigma2.beta.a - 1)
    sigma2.beta.var = sigma2.beta.b^2/( (sigma2.beta.a - 1)^2 * (sigma2.beta.a - 2))
    sigma2.beta.std = sqrt(sigma2.beta.var)
    sigma2.beta.95cilb = qinvgamma(c( .025), shape=sigma2.beta.a, rate =sigma2.beta.b)
    sigma2.beta.95ciub = qinvgamma(c( .975), shape=sigma2.beta.a, rate =sigma2.beta.b)
    sigma2.beta = cbind(mode = sigma2.beta.mode, std = sigma2.beta.std,
                        CILB = sigma2.beta.95cilb, CIUB = sigma2.beta.95ciub)
    row.names(sigma2.beta)=c("sigma2.beta1","sigma2.beta2","sigma2.beta3")
    
    # R0
    R01 = beta[1,]/gamma
    R02 = beta[2,]/gamma
    R03 = beta[3,]/gamma
    R0 = rbind(R01, R02,R03)
    R0[,2] = sqrt((beta.mu / gamma.mu)^2 * (beta.sigma2 / beta.mu^2 + gamma.sigma2 / gamma.mu^2))
    R0[,3] = R0[,1] - 1.96 * R0[,2]
    R0[,4] = R0[,1] + 1.96 * R0[,2]
    row.names(R0) = c("R01", "R02","R03")
    
    # combine parameters
    parameter = rbind(gamma, sigma2.gamma, beta, sigma2.beta, R0)
    return(round(parameter, 6))
    
  }
  

}

#' addTltoData
#' Especially for US data
#'
#' @param city_data data frames for each city, which is data[[l]]
addTltoData <- function(city_data){
  CI <-  city_data$infected
  n <- length(CI)
  i=0
  # consider case when there's no input WV us[[37]]
  while ( (i <= (n-1) && (CI[i+1]==0)  )){
    i=i+1
  }
  if (i == n){
    #cat("This city is safe by now, without any cases confirmed.")
    tl = NA
  } else{
    i = i+1
    tl = rep(i, n)
  }
  city_data$tl = tl
  return(city_data)
}


#' getbetaforOneCity
#' Check city's Zlts/Zltr for debugging purpose
#'
#' e.g. data_cn[[37]] Yichang
#' e.g. data_us[[57]] WV
#'
#' @param data data as a list with L data frames for each city
#' @param l  integer, identifying which city or state to see
#' @param t0 Use data until A_{t0-1} and S_{t0}
#' @param lag shifted time between data and self defined state variable
#' @param MA_index = 0.4 central weight for smooth moving average method
#' @param t_start = 0 vector of integers
getbetaforOneCity <-function(data, l, t0, lag = 8, MA_index = 0.4, t_start = 1){
  f1 = f2 = 1
  Xl = addTltoData(data[[l]])
  tl = Xl$tl[1];tl

  Xl_shifted = shift_state(Xl, lag = lag, MA = MA_index)
  XS = Xl_shifted$XS
  XI = Xl_shifted$XI
  XR = Xl_shifted$XR
  A =  Xl_shifted$A

  for (t in (tl + t_start):(t0-1)){
    Xti = XI[t]
    Xtr = XR[t]
    Xts = XS[t]

    Xti_p1 = XI[t+1]
    Xtr_p1 = XR[t+1]
    Xts_p1 = XS[t+1]

    Zltr =  (Xtr_p1 - Xtr)/(f2*Xti)
    Zlts =  -(Xts_p1 - Xts)/(f1*Xti*Xts)

    if (Xti > 1e-12 ){
      i = A[t]
      cat("l:",l,"t",t,": Zltr=",Zltr,"  .","Zlts=", Zlts, "action=", i, "total:", XR[t] * Xl$population[1], "\n")
    }
  }
}
