# cat(a / b)



# R0 = rep(0,J)
# for (i in 1:3) {
#   cat(beta.pos[[i]]$coefficients,"\n")
#   R0[i] = beta.pos[[i]]$coefficients[2] / gamma.pos$coefficients[2]
# }
# cat(R0)
# for (i in 1:3) {
#   cat(  beta.pos[[i]]$coefficients, "\n")
# }

# glm(y ~ 0 + x, data = gamma_data, family = poisson(link = "identity"))
# gamma.pos <- stan_glm(y ~ x, data = gamma_data, family = poisson(link = "identity"), 
#                       prior = normal(mu.gamma, sigma.gamma, autoscale=FALSE),
#                       prior_intercept = normal(0, 0.00001, autoscale=FALSE), adapt_delta = 0.9999,
#                       algorithm = "sampling", 
#                       seed = 1) #  optimizing
# gamma.pos$coefficients
# 
# beta.pos  = list()
# beta.MAP = rep(0,J)
# a = glm(y ~ 0 + x, data = data_beta, family = poisson(link = "identity"))
# cat(a$coefficients)

# stan_glm1 <- stan_glm(y ~ x, data = data_beta, family = poisson(link = "identity"), 
#                       prior = normal(mu.beta[i], sigma.beta[i], autoscale=FALSE),
#                       prior_intercept = normal(0, 0.00001, autoscale=FALSE), adapt_delta = 0.9999, 
#                       algorithm = "sampling", 
#                       seed = 1) #  
#  beta.pos[[i]] = stan_glm1
#  beta.MAP[i] = stan_glm1$coefficients[2]
# cat(stan_glm1$coefficients)


# posterior_interval(beta.pos[[i]], prob = 0.9, pars = "x")
# a = as.matrix(stan_glm1)[,2]

# gamma
# gamma.mode = MB / MA
# gamma.v = (2*aR + Gt0)
# gamma.mu = MB / MA
# gamma.sigma2 = (2*MA*(bR+MC)-MB^2)/(MA^2*(gamma.v))
# gamma.var = (gamma.v / (gamma.v-2) )* gamma.sigma2
# gamma.std = sqrt(gamma.var)
# gamma.95cilb = qt(c(.025), df=gamma.v) * sqrt(gamma.sigma2) + gamma.mu
# gamma.95ciub = qt(c(.975), df=gamma.v) * sqrt(gamma.sigma2) + gamma.mu
# gamma = cbind(mode = gamma.mode, std = gamma.std, CILB = gamma.95cilb, CIUB = gamma.95ciub)
# row.names(gamma) = "gamma"
# 
# # beta
# beta.mode = MBI / MAI
# beta.v = (2*aS + GtI)
# beta.mu = MBI / MAI
# beta.sigma2 = (2*MAI*(bS+MCI)-MBI^2)/(MAI^2*(beta.v))
# beta.var = (beta.v) / (beta.v-2) * beta.sigma2
# beta.std = sqrt(beta.var)
# beta.95cilb = qt(c(.025), df=beta.v) * sqrt(beta.sigma2) + beta.mu
# beta.95ciub = qt(c(.975), df=beta.v) * sqrt(beta.sigma2) + beta.mu
# beta = cbind(mode = beta.mode, std = beta.std, CILB = beta.95cilb, CIUB = beta.95ciub)
# row.names(beta)=c("beta1","beta2","beta3")
# 
# 
# # R0
# R01 = beta[1,]/gamma
# R02 = beta[2,]/gamma
# R03 = beta[3,]/gamma
# R0 = rbind(R01, R02,R03)
# R0[,2] = sqrt((beta.mu / gamma.mu)^2 * (beta.sigma2 / beta.mu^2 + gamma.sigma2 / gamma.mu^2))
# R0[,3] = R0[,1] - 1.96 * R0[,2]
# R0[,4] = R0[,1] + 1.96 * R0[,2]
# row.names(R0) = c("R01", "R02","R03")
# 
# # combine parameters
# parameter = rbind(gamma, beta, R0)


# Func: old -------------------------------------------------------------------------


#' #' getSIRMAP
#' #' Use observations until A_{t0-1} and X_{t0} to train the parameter theta.
#' #'
#' #' @param data data as a list with L data frames for each city, data$infected, $suspected, $removed, $action, $population
#' #' @param t0 Use data until A_{t0-1} and S_{t0} to train the parameters
#' #' @param lag shifted time between data and self defined state variable
#' #' @param J integer, number of action levels
#' #' @param u_2 = 0.5 prior for relative reansmission rate of action 1 to action 0
#' #' @param u_3 = 0.1 prior for relative reansmission rate of action 2 to action 0
#' #' @param region_index vector of integers, indicating which region you are gonna select for estimation
#' #' @param MA = 0.4 central weight for smooth moving average method
#' #' @param echo boolean variable indicating if the zlts/zltr should be output during the process
#' #' @param t_start = 0 vector of integers
#' getSIRMAP <- function(data=data_cn, t0=25, lag=8, J=3, u_2 = 0.5, u_3 = 0.1, region_index=c(1:50), MA_index = 0.4, echo = F, t_start = 0){
#'   ## Use data until A_{t0-1} and S_{t0}
#'   # t0: t0 + lag should be less than max length (42 for now)
#'   t_max = dim(data[[1]])[1]
#'   if((t0) > t_max){
#'     stop(paste("The input t0 should be an integer from 1+tl+t_start to", t_max,".\n"))
#'   }
#' 
#'   # empty vector initialization
#'   # f1 <- rep(1,N)
#'   # f2 <- rep(1,N)
#'   MA <- c(0)
#'   MB <- c(0)
#'   MC <- c(0)
#'   MAI <- rep(0,J)
#'   MBI <- rep(0,J)
#'   MCI <- rep(0,J)
#'   # Parameter Initialization prior
#' 
#'   J = 3
#'   aR = 3
#'   bR = 2 * 1e-10
#'   aS = rep(3, J)
#'   bS = rep(2, J) * 1e-10
#'   mu.gamma = .0117
#'   sigma.gamma = .1
#'   mu.beta = c(3.15, 3.15 * u_2, 3.15 * u_3) * mu.gamma
#'   sigma.beta = rep(0.5,J) # 0.33 based on Peter_Song
#' 
#' 
#'   # load function f(X_l)
#'   f1_func(data,location=1) # range(1,50)
#'   f2_func(data,location=5)
#' 
#'   if (lag==0){
#'     for (l in region_index){
#'       f1 = f1_func(data, l)
#'       f2 = f2_func(data, l)
#'       Xl = addTltoData(data[[l]][1:(t0),])
#'       tl = Xl$tl[1]
#'       if(is.na(tl)){
#'         next
#'       }
#'       XI = Xl$infected
#'       XR = Xl$remove
#'       XS = 1 - XI - XR
#'       for (t in (tl + t_start):(t0-1)){
#'          if (XI[t] > 0 ){#  & Zltr > 0
#'           # normal version without shift
#'           Zltr =  (XR[t+1] - XR[t])/(f2*XI[t])
#'           MA = MA + 1 # counter
#'           MB = MB + Zltr
#'           MC = MC + Zltr^2
#'           i = Xl$action[t]+1
#' 
#'           Zlts =  -(XS[t+1] - XS[t])/(f1*XI[t]*XS[t])
#'           MAI[i] = MAI[i] + 1 # counter
#'           MBI[i] = MBI[i] + Zlts
#'           MCI[i] = MCI[i] + Zlts^2
#'           if(echo)cat("l:",l,"t",t,": Zltr=",Zltr,"  .","Zlts=", Zlts, "action=", i,"\n")
#'          }
#'       }
#'     }
#'   } else{
#'     # shifted version
#'     for (l in region_index){
#'       f1 = f1_func(data, l)
#'       f2 = f2_func(data, l)
#'       Xl = addTltoData(data[[l]])
#'       tl = Xl$tl[1]
#' 
#'       # If there's no available data, then skip this city
#'       if(is.na(tl)){
#'         next
#'       }
#' 
#'       # 9:50 shift to 1:42
#'       Xl_shifted = shift_state(Xl, lag = lag, MA = MA_index)
#'       XS = Xl_shifted$XS
#'       XI = Xl_shifted$XI
#'       XR = Xl_shifted$XR
#'       A =  Xl_shifted$A
#' 
#'       for (t in (tl + t_start):(t0-1)){
#' 
#'         Xti = XI[t]
#'         Xtr = XR[t]
#'         Xts = XS[t]
#' 
#'         Xti_p1 = XI[t+1]
#'         Xtr_p1 = XR[t+1]
#'         Xts_p1 = XS[t+1]
#' 
#'         if ( Xti > 1e-11 ){#  & Zltr > 0
#'           Zltr =  (Xtr_p1 - Xtr)/(f2*Xti)
#'           Zlts =  -(Xts_p1 - Xts)/(f1*Xti*Xts)
#' 
#'           MA = MA + 1 # counter
#'           MB = MB + Zltr
#'           MC = MC + Zltr^2
#'           i = A[t]
#' 
#'           MAI[i] = MAI[i] + 1 # counter
#'           MBI[i] = MBI[i] + Zlts
#'           MCI[i] = MCI[i] + Zlts^2
#'           if(echo)cat("l:",l,"t",t,"xti",Xti,": Zltr=",Zltr,"  .","Zlts=", Zlts, "action=", i,"\n")
#' 
#'         }
#'       }
#'     }
#'   }
#' 
#'   Gt0 = MA
#'   GtI = MAI
#'   cat("GTI",sum(GtI))
#'   # Posterior mean
#'   MA = MA + 1/(sigma.gamma^2) # counter
#'   MB = MB + mu.gamma/sigma.gamma
#'   MC = MC + (mu.gamma/sigma.gamma)^2
#'   MAI = MAI + 1/(sigma.beta^2)
#'   MBI = MBI + mu.beta/sigma.beta
#'   MCI = MCI + (mu.beta/sigma.beta)^2
#' 
#'   # gamma
#'   gamma.mode = MB / MA
#'   gamma.v = (2*aR + Gt0)
#'   gamma.mu = MB / MA
#'   gamma.sigma2 = (2*MA*(bR+MC)-MB^2)/(MA^2*(gamma.v))
#'   gamma.var = (gamma.v / (gamma.v-2) )* gamma.sigma2
#'   gamma.std = sqrt(gamma.var)
#'   gamma.95cilb = qt(c(.025), df=gamma.v) * sqrt(gamma.sigma2) + gamma.mu
#'   gamma.95ciub = qt(c(.975), df=gamma.v) * sqrt(gamma.sigma2) + gamma.mu
#'   gamma = cbind(mode = gamma.mode, std = gamma.std, CILB = gamma.95cilb, CIUB = gamma.95ciub)
#'   row.names(gamma) = "gamma"
#' 
#'   # beta
#'   beta.mode = MBI / MAI
#'   beta.v = (2*aS + GtI)
#'   beta.mu = MBI / MAI
#'   beta.sigma2 = (2*MAI*(bS+MCI)-MBI^2)/(MAI^2*(beta.v))
#'   beta.var = (beta.v) / (beta.v-2) * beta.sigma2
#'   beta.std = sqrt(beta.var)
#'   beta.95cilb = qt(c(.025), df=beta.v) * sqrt(beta.sigma2) + beta.mu
#'   beta.95ciub = qt(c(.975), df=beta.v) * sqrt(beta.sigma2) + beta.mu
#'   beta = cbind(mode = beta.mode, std = beta.std, CILB = beta.95cilb, CIUB = beta.95ciub)
#'   row.names(beta)=c("beta1","beta2","beta3")
#' 
#'   # sigma2.gamma
#'   sigma2.gamma.a = aR + Gt0/2
#'   sigma2.gamma.b = bR + MC/2-MB^2/(2*MA)
#'   sigma2.gamma.mode = sigma2.gamma.b/(sigma2.gamma.a + 1)
#'   sigma2.gamma.mu = sigma2.gamma.b/(sigma2.gamma.a - 1)
#'   sigma2.gamma.var = sigma2.gamma.b^2/( (sigma2.gamma.a - 1)^2 * (sigma2.gamma.a - 2))
#'   sigma2.gamma.std = sqrt(sigma2.gamma.var)
#'   sigma2.gamma.95cilb = qinvgamma(c( .025), shape=sigma2.gamma.a, rate =sigma2.gamma.b)
#'   sigma2.gamma.95ciub = qinvgamma(c( .975), shape=sigma2.gamma.a, rate =sigma2.gamma.b)
#'   sigma2.gamma = cbind(mode = sigma2.gamma.mode, std = sigma2.gamma.std,
#'                        CILB = sigma2.gamma.95cilb, CIUB = sigma2.gamma.95ciub)
#'   row.names(sigma2.gamma) = "sigma2.gamma"
#' 
#' 
#' 
#'   # sigma2.beta
#'   sigma2.beta.a = aS + GtI/2
#'   sigma2.beta.b = bS + MCI/2-MBI^2/(2*MAI)
#'   sigma2.beta.mode = sigma2.beta.b/(sigma2.beta.a + 1)
#'   sigma2.beta.mu = sigma2.beta.b/(sigma2.beta.a - 1)
#'   sigma2.beta.var = sigma2.beta.b^2/( (sigma2.beta.a - 1)^2 * (sigma2.beta.a - 2))
#'   sigma2.beta.std = sqrt(sigma2.beta.var)
#'   sigma2.beta.95cilb = qinvgamma(c( .025), shape=sigma2.beta.a, rate =sigma2.beta.b)
#'   sigma2.beta.95ciub = qinvgamma(c( .975), shape=sigma2.beta.a, rate =sigma2.beta.b)
#'   sigma2.beta = cbind(mode = sigma2.beta.mode, std = sigma2.beta.std,
#'                       CILB = sigma2.beta.95cilb, CIUB = sigma2.beta.95ciub)
#'   row.names(sigma2.beta)=c("sigma2.beta1","sigma2.beta2","sigma2.beta3")
#' 
#'   # R0
#'   R01 = beta[1,]/gamma
#'   R02 = beta[2,]/gamma
#'   R03 = beta[3,]/gamma
#'   R0 = rbind(R01, R02,R03)
#'   R0[,2] = sqrt((beta.mu / gamma.mu)^2 * (beta.sigma2 / beta.mu^2 + gamma.sigma2 / gamma.mu^2))
#'   R0[,3] = R0[,1] - 1.96 * R0[,2]
#'   R0[,4] = R0[,1] + 1.96 * R0[,2]
#'   row.names(R0) = c("R01", "R02","R03")
#' 
#'   # combine parameters
#'   parameter = rbind(gamma, sigma2.gamma, beta, sigma2.beta, R0)
#'   return(round(parameter, 6))
#' }
#' 


if (lag==0){
  for (l in region_index){
    f1 = f1_func(data, l)
    f2 = f2_func(data, l)
    Xl = addTltoData(data[[l]][1:(t0),])
    tl = Xl$tl[1]
    if(is.na(tl)){
      next
    }
    XI = Xl$infected
    XR = Xl$remove
    XS = 1 - XI - XR
    for (t in (tl + t_start):(t0-1)){
      if (XI[t] > 0 ){#  & Zltr > 0
        # normal version without shift
        Zltr =  (XR[t+1] - XR[t])/(f2*XI[t])
        MA = MA + 1 # counter
        MB = MB + Zltr
        MC = MC + Zltr^2
        i = Xl$action[t]+1
        
        Zlts =  -(XS[t+1] - XS[t])/(f1*XI[t]*XS[t])
        MAI[i] = MAI[i] + 1 # counter
        MBI[i] = MBI[i] + Zlts
        MCI[i] = MCI[i] + Zlts^2
        if(echo)cat("l:",l,"t",t,": Zltr=",Zltr,"  .","Zlts=", Zlts, "action=", i,"\n")
      }
    }
  }
}