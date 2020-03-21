# Transition functions ----------------------------------------------------

# Y1 = g(X_lt1_old, A1, X_lt1, X_l, w)
g <-function(X_lt_old, A, X_lt, X_l, w){
  return()
}

#' @param X_lt (S, I, R)
g_R <-function(X_lt_old, X_lt){
  return(-sum(X_lt[1:2] - X_lt_old[1:2]))
}

g_C <-function(){
  return()
}


#' f
#' State transition function, sampled based on the SIR model with current parameter estimates
#'
#' @param theta (sigma_R, gamma, sigma_S, beta) or (gamma, beta)
#' @param A  A_{l,t} among {1,...,J}
#' @param XR_1 if not NA, it indicts the XR at next time is observable and should be respected.

#' @export nextstate  c(XS=XS_new, XI=XI_new, XR=XR_new)
#' @export poisson  whether or not to use poisson

f <- function(X_lt, A, theta, f1 = NA, X_l = NA, cut_0 = T, XR_1 = NA, poisson = T){
  # get current state variables and parameters
  J = 3
  XS = X_lt[1]
  XI = X_lt[2]
  XR = X_lt[3]
  popu = XS + XI + XR
  
  
  f1 = f2 = 1
  # no f1, f2 yet
  if(poisson){# Poisson: counts
    gamma = theta[1]
    beta = theta[1 + A]
    if (is.na(XR_1) ){
      XS_new = XS -  rpois(1, beta * f1 * XS * XI / popu)
      XR_new = XR + rbinom(1, XI, gamma * f2)
      XI_new = popu - XS_new - XR_new
    }else{# next XR is observable
      XS_new = XS -  rpois(1, beta * f1 * XS * XI / popu)
      XR_new = XR
      XI_new = popu - XS_new - XR_new
    }
  }else{# Gaussian: proportions
    gamma = theta[1]
    sigma_R = sqrt(theta[2])
    beta = theta[3 + A - 1]
    sigma_S = sqrt(theta[3 + J + A - 1])
    if(cut_0 & XI <=0){
      # stop spread
      eS = eR = 0
      XI_new = 0
      XR_new = XR
      XS_new = 1 - XR_new - XI_new
    }else{
      eS_init = rnorm(1, 0, sigma_S) 
      eS = eS_init *  XS * XI
      eR_init = rnorm(1, 0, sigma_R) 
      eR = eR_init * XI
      if (is.na(XR_1) ){
        # next XR is observable
        XS_new = XS - beta * XS * XI + eS
        XI_new = XI + beta * XS * XI - gamma * XI - ( eS + eR)
        XR_new = XR + gamma * XI + eR
      }else{
        # normal sampling
        XR_new = XR_1
        XI_new = XI + beta * XS * XI - eS - (XR_new - XR)
        XS_new = 1 - XR_new - XI_new
      }
      
    }
    if(XR_new < XR){
      cat("bugs!:", X_lt,"|||", c(XS=XS_new, XI=XI_new, XR=XR_new), "|||", c(eS_init, eS, eR_init, eR),"\n")
    }
  }
  return(c(XS=XS_new, XI=XI_new, XR=XR_new))
}


# Helper  funuctions ------------------------------------------------------

#' Smooth moving average for a series
SMA <- function(series, weight_center){
  l = length(series)
  new_series = c(series[1])
  w = (1 - weight_center) / 2
  for (i in 2:(l-1)) {
    new_series = c(new_series, w * series[i - 1] + w * series[i + 1] + weight_center * series[i])
  }
  return(c(new_series, series[l]))
}


#' shift_state
#'
#' @param Xl n by p, the original data for l^th region.
#' @param lag =8 shifted time length.
#' @param MA = NA /.4,  whether or not to use moving average for state variables
#'                 and the weight of the middle point.
#'
#' @export Xl_shifted a data frame with (n+lag) by p. The last lag elements of XI and XS are NAs.
#' @export XS 1 - XI - XR
#' @export XI shifted infectious population ~= future eight days increasing infectious population
#' @export XR removed population ~= cumulative confirmed cases
#' @export A 1 to J.
#' Xl_s() = shift_state(Xl) # Xl:1/24-3/5 to Xl_s:1/16-2/26
shift_state <- function(Xl, lag = 8, MA = NA, popu = NA){
  ###
  # MA:
  dates = Xl$date
  beforedates = rev(seq(as.Date(dates[1]),  by="-1 day", length.out = lag+1))[1:lag]
  date = c(beforedates, dates)
  I_confirmed = Xl$infected
  if (! is.na(MA)) {
    I_confirmed = SMA(I_confirmed, 0.4)
  }
  I_confirmed = c(rep(0, lag), I_confirmed)
  XI = data.table::shift(I_confirmed, n = lag, fill = NA, type="lead") - I_confirmed
  XR = I_confirmed
  if (is.na(popu)) {
    XS = 1 - XI - XR
  }else{
    XS = popu - XI - XR
  }
  
  #action_shift = data.table::shift(Xl$action, n = lag, fill = NA, type="lead")
  A = c(rep(0, lag), Xl$action) + 1
  
  #if(times_population)return(list(XI, XR, XS, A))
  shift_data <- data.frame(date, XS, XI, XR, A)
  return(shift_data)
}


## newInfect: calculate new infected number as XR_t - XR_{t-1}
newInfect <- function(removed){
  r = removed - data.table::shift(removed, n = 1, fill = 0, type="lag")
  r[1] = r[2]
  return(r)
}

# Plot functions ----------------------------------------------------------


arrangeGgplot <- function(ggplots, n, p){
  if ( (n%%2) == 0){
    for (i in 1:(n/p)){
      grid.newpage()
      pushViewport(viewport(layout = grid.layout(3,2)))
      print(ggplots[[p*i-(p-1)]], vp = viewport(layout.pos.row = 1, layout.pos.col = 1))
      print(ggplots[[p*i-(p-2)]], vp = viewport(layout.pos.row = 2, layout.pos.col = 1))
      print(ggplots[[p*i-(p-3)]], vp = viewport(layout.pos.row = 3, layout.pos.col = 1))
      print(ggplots[[p*i-(p-4)]], vp = viewport(layout.pos.row = 1, layout.pos.col = 2))
      print(ggplots[[p*i-(p-5)]], vp = viewport(layout.pos.row = 2, layout.pos.col = 2))
      print(ggplots[[p*i-(p-6)]], vp = viewport(layout.pos.row = 3, layout.pos.col = 2))
    }
  }
}



plotR0 <- function(post="data/result/SIR_post_mode_origin.csv", ylimit=c(0,10), maintitle="title"){
  params <- read.csv(file=post,header = T)
  date <- params$X
  colours = c("black","turquoise3", "darkgoldenrod1","orangered")
  xlab = as.Date(as.character(date), "%Y-%m-%d");head(xlab)
  max = max( params$R0.post1, params$R0.post2, params$R0.post3 )
  plot(xlab,ylim=ylimit, params$R0.post1,ylab="R0.post.mean",
       type="l", main=maintitle,
       xlab="date", col=colours[2])
  lines(xlab, params$R0.post2, type="l", col=colours[3])
  lines(xlab, params$R0.post3, col=colours[4])
  legend("topright",8,c("action=0","action=1","action=2"),cex=.7 ,lty=1, col=colours[c(2:4)])
}




plotPrediction_origin <- function(XSest, XIest, XRest, l, t_begin = 31, t_end = 42, poisson = T){
  f1 = f1_func(data, l)
  f2 = f2_func(data, l)
  Xl = addTltoData(data[[l]])
  # population?
  population = Xl[1,10]
  tl = Xl$tl[1]
  
  Xl_shifted = shift_state(Xl)
  if (poisson) {
    XI = Xl_shifted[[1]] 
    XR = Xl_shifted[[2]] 
    XS = Xl_shifted[[3]] 
  }else{
    XI = Xl_shifted[[1]] * population
    XR = Xl_shifted[[2]] * population
    XS = Xl_shifted[[3]] * population
  }
  A =  Xl_shifted[[4]]
  X_lts = cbind(XS, XI, XR)
  
  colours = c("black","turquoise3", "darkgoldenrod1","orangered")
  
  par(mfrow=c(1,3),mgp = c(0, 0.5, 0), mar=c(3, 4, 3, 4), oma=c(0,0,2,0))
  XIest
  xlab = as.Date(as.character(Xl$dates), "%Y-%m-%d");head(xlab)
  plot(xlab[c(1:t_end)], c(XI[1:t_end]) ,type="l",xlab = "", ylab="XI",main="Prediction of XI in Wuhan")
  lines(xlab[c(t_begin:t_end)], XIest[,2], col=colours[4])
  lines(xlab[c(t_begin:t_end)], XIest[,1], col=colours[2])
  lines(xlab[c(t_begin:t_end)], XIest[,3],col=colours[2])
  sort(XIest[,3]-XIest[,1], decreasing=FALSE)
  legend("topright",8,c("data","estimate","95%PI"),cex=.9 ,lty=1, col=colours[c(1,4,2)])
  
  
  XRest
  xlab = as.Date(as.character(Xl$dates), "%Y-%m-%d");head(xlab)
  plot(xlab[c(1:t_end)], c(XR[1:t_end]) ,type="l", xlab = "",ylab="XI",
       main="Prediction of XR in Wuhan")
  lines(xlab[c(t_begin:t_end)], XRest[,2], col=colours[4])
  lines(xlab[c(t_begin:t_end)], XRest[,1], col=colours[2])
  lines(xlab[c(t_begin:t_end)], XRest[,3],col=colours[2])
  legend("bottomright",8,c("data","estimate","95%PI"),cex=.9 ,lty=1, col=colours[c(1,4,2)])
  
  XSest
  range( range(XSest),range(XS) )
  xlab = as.Date(as.character(Xl$dates), "%Y-%m-%d");head(xlab)
  plot(xlab[c(1:t_end)], c(XS[1:t_end]) ,type="l", xlab = "",ylab="XI",
       main="Prediction of XS in Wuhan")
  lines(xlab[c(t_begin:t_end)], XSest[,2], col=colours[4])
  lines(xlab[c(t_begin:t_end)], XSest[,1], col=colours[2])
  lines(xlab[c(t_begin:t_end)], XSest[,3],col=colours[2])
  legend("topright",8,c("data","estimate","95%PI"),cex=.9 ,lty=1, col=colours[c(1,4,2)])
  #mtext("Predict by Wuhan's first 30 days data", outer = TRUE, cex = 1)
  mtext("Predict by all cities' first 30 days data", outer = TRUE, cex = 1)
  
}

# MAP ---------------------------------------------------------------------




MAP <-function(data, prior.mu, prior.sigma, low = 0.001, high = 1, is_gamma = F){
  if(is_gamma){#binomial
    logL <-function(data, slope){
      # one_den <-function(vec){
      #   return(dbinom(x = vec[2], size = vec[1], prob = .5 , log = T))
      # }
      # apply(a, 1 ,one_den)
      return(sum(dbinom(x = data[,2], size = data[,1], prob = slope , log = T)))
    }
  }else{
    logL <-function(data, slope){
      return(sum(dpois(data[,2], slope * data[,1], log = T)))
    }
  }
  
  
  prior <- function(x, prior.mu, prior.sigma){
    return(dnorm(x, prior.mu,prior.sigma, log = T))
  }
  
  posterior <-function(data, slope, prior.mu, prior.sigma){
    return(logL(data, slope) + prior(slope, prior.mu, prior.sigma))
  }
  
  objective <- function(slope){
    return(posterior(data, slope, prior.mu, prior.sigma))
  }
  return(optimize(objective, c(low, high), maximum = T))
}


# Data Preprocessing ------------------------------------------------------



mergeUSdata <- function(data_us){
  merge_data  = data_us[[1]][,c(1, 8:12)]
  L = length(data_us)
  t = dim(data_us[[1]])[2]
  names(merge_data)
  # population
  (population = unlist( sapply(data_us, "[",1,'population') )  )
  merge_population = sum(population)
  merge_data$population = merge_population
  merge_data$population
  
  # infected
  infect_seperate = as.data.frame(sapply(data_us, "[",,'infected'))
  head(infect_seperate)
  infect_seperate = t( t(infect_seperate) * population)
  merge_data$infected = apply(infect_seperate, 1, sum)/merge_population
  
  # removed
  remove_seperate = as.data.frame(sapply(data_us, "[",,'remove'))
  head(remove_seperate)
  remove_seperate = t( t(remove_seperate) * population)
  merge_data$remove = apply(remove_seperate, 1, sum)/merge_population
  
  # suspected
  # remove_seperate = as.data.frame(sapply(data_us, "[",,'suspect'))
  # head(remove_seperate)
  # remove_seperate = t( t(remove_seperate) * population)
  # var1 = apply(remove_seperate, 1, sum)/merge_population
  # var2 =  1 - merge_data$infected - merge_data$remove
  # all.equal(var1, var2)
  merge_data$suspect = 1 - merge_data$infected - merge_data$remove
  head(merge_data)
  # action? need? might don't need
  data_us[[L+1]] = merge_data
  
  return(data_us)
}
