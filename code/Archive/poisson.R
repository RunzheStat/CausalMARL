

b = MAP(gamma_data, prior.mu = .0117, prior.sigma = 0.1, low = 0, high = 1)$maximum


a = rep(0,3)
for (i in 1:J) {
  
  data_beta = do.call(rbind, beta_data[[i]])
  data_beta = as.data.frame(data_beta)
  names(data_beta)  = c("x", "y")
  a[i] = MAP(data_beta, prior.mu = 0.03, prior.sigma = 0.5, low = 0, high = 1)$maximum
}

cat(a / b)


# 
# 
# posterior(gamma_data,0.1, 0.2, .1)
# posterior(gamma_data,0.2, 0.2, .1)
# 
# prior.mu = 10
# prior.sigma = 0.1

# library(rjags)
# 
# 
# 
# N <- 1000
# beta0 <- 1  # intercept
# beta1 <- 1  # slope
# x <- abs(rnorm(n=N))  # standard Normal predictor
# mu <- beta0*1 + beta1*x  # linear predictor function
# # lambda <- exp(mu)  # CEF
# lambda <- exp(mu)  # CEF
# y <- rpois(n=N, lambda=lambda)  # Poisson DV
# dat <- data.frame(x,y)  
# 
# forJags <- list(X= dat$x,# cbind(1,dat$x),  # predictors
#                 # bind(1,dat$x)
#                 y=dat$y,  # DV
#                 N=N,  # sample size
#                 mu.beta=rep(0,1),  # priors centered on 0
#                 tau.beta=diag(.0001,1))  # diffuse priors
# 
# jagsmodel <- jags.model(file="/Users/mac/Desktop/pois.bug",  # compile model
#                         data=forJags)
# 

