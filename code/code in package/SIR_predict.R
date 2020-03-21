
#' validation
#' Make prediction based on given theta
#'
#' @param data  data_cn or data_us
#' @param theta parameter estimated
#' @param l index of city between 1:length(data)
#' @param t_begin = 21 the begin date for prediction, 0 < t_begin <= data_end
#' @param t_end = 42 the end day for prediction date, t_begin<=t_end, no upper limit.
#' @param rep=10000 number of replicates to get the MC PI for future predictions. The larger, the smoother.
#' @param alpha = 0.05, significance level = 100(1-alpha)%
#' @param actions = NA or a vector of action levels at {1, 2, ..., J} with length = t_end-t_begin+1,
#' @param MA = 0.4 central weight for smooth moving average method, between 0 to 1
#'
#' @export newstate = list(XSest=XSest * population, XIest = XIest * population, XRest = XRest * population, XNest = XNest * population)
#'
prediction <- function(data, theta, l, t_begin = 26, t_end = 42, rep=1000, alpha = 0.9, actions = NA, MA_index = NA, echo=FALSE, poisson = T){
  ####### Based on estimated parameter until t_begin, construct PI until t_end, following observed actions or specified actions
  ### Start from S_{t0}, using A_{t0}, ..., predict S_{t0+1}, ....
  ## Args
  #   t_begin: = t0 + 1
  #   actions: if "NA", then use observed  actions to do validation; otherwise, use specified actions to evaluate policies
  #
  ## Output: the estimated states for (t_begin, t_end]
  #
  L = length(data)
  if((l<0) | (l>L)){
    stop(paste("The input city index l should be an integer from 1 to", L,".\n"))
  }
  if((t_begin<0)){
    stop(paste("Please input a positive t_begin larger than 0.\n"))
  }
  if((t_end< t_begin)){
    stop(paste("Please input a t_end no less than t_begin.\n"))
  }
  if ( (sum(is.na(actions))==0) && (length(actions) != t_end-t_begin+1) ){
    stop(paste("Please input actions with length,", t_end-t_begin+1,".\n"))
  }
  ## preprocess data
  f1 = f1_func(data, l)
  f2 = f2_func(data, l)
  Xl = addTltoData(data[[l]])
  tl = Xl$tl[1]
  population = Xl$population[1]
  Xl_shifted = shift_state(Xl, MA = MA_index, popu = population)
  XS = Xl_shifted$XS
  XI = Xl_shifted$XI
  XR = Xl_shifted$XR
  A =  Xl_shifted$A
  X_lts = cbind(XS, XI, XR)


  if (t_end > dim(Xl)[1]){
    X_lts = rbind(X_lts, matrix(0, nrow=(t_end-dim(Xl)[1]+1), ncol=3  ))
  }

  ## Initialization
  XIest = matrix(0,nrow = c(t_end-t_begin+1), ncol=3)
  rownames(XIest) <- c(t_begin:t_end)
  XNest = XSest = XRest = XIest
  XNtraj = XStraj = XRtraj = XItraj = matrix(0, nrow = rep, ncol=c(t_end-t_begin+1))

  # Generate trajectories
  count = 1
  for (r in 1:rep){
    for (t in t_begin:t_end){
      if(count < 8){# XR is observable
        if(is.na(actions[1])){
          #X_lts[t,] = f(X_lts[t-1,], A[t-1], theta, XR_1 = X_lts[t,3])
          X_lts[t,] = f(X_lts[t-1,], A[t-1], theta)
          t = t + 1
          if (echo==T){cat("X_lts[t,]",X_lts[t,])}
        }else{
          #X_lts[t,] = f(X_lts[t-1,], actions[t - t_begin + 1], theta, XR_1 = X_lts[t,3])
          X_lts[t,] = f(X_lts[t-1,], actions[t - t_begin + 1], theta)
        }
      }else{# XR is not observable
        if(is.na(actions)[1]){
          X_lts[t,] = f(X_lts[t-1,], A[t-1], theta)
        }else{
          X_lts[t,] = f(X_lts[t-1,], actions[t - t_begin + 1], theta)
        }
      }
      count = count + 1
    }
    XStraj[r,] = X_lts[t_begin:t_end,1]
    XItraj[r,] = X_lts[t_begin:t_end,2]
    XRtraj[r,] = X_lts[t_begin:t_end,3]
    XNtraj[r,] = newInfect(X_lts[t_begin:t_end,3])
  }
  colMeans(XStraj)

  # Calculate means and PIs
  XSest = t( apply(XStraj, 2, quantile, c(alpha / 2, .5, 1 - alpha / 2)) )
  XSest[,2] = colMeans(XStraj); colnames(XSest)[2] = "mean"
  XIest = t( apply(XItraj, 2, quantile, c(alpha / 2, .5, 1 - alpha / 2)) )
  XIest[,2] = colMeans(XItraj); colnames(XIest)[2] = "mean"
  XRest = t( apply(XRtraj, 2, quantile, c(alpha / 2, .5, 1 - alpha / 2)) )
  XRest[,2] = colMeans(XRtraj); colnames(XRest)[2] = "mean"
  XNest = t( apply(XNtraj, 2, quantile, c(alpha / 2, .5, 1 - alpha / 2)) )
  XNest[,2] = colMeans(XNtraj); colnames(XNest)[2] = "mean"

  if(poisson){
    newstate = list(XSest=XSest , XIest = XIest , XRest = XRest , XNest = XNest )
  }else{
    newstate = list(XSest=XSest * population, XIest = XIest * population, XRest = XRest * population, XNest = XNest * population)
  }
  return(newstate)
}



#' plotValidation
#' Make prediction based on given theta before t_begin
#'
#' @import newstate
#'
#' @param data  data_cn or data_us
#' @param newstate export from prediction(t_begin = 31, t_end = 55)
#' @param l index of city or state
#' @param t_begin = 31 the start of the prediction date. t_begin > 0
#' @param t_end = 55 the end of the prediction date, t_begin<t_end<length(data[[1]][,1]).
#' @param MA = 0.4 central weight for smooth moving average method, between 0 to 1
#' @param nrow=3 plot parameter
#' @param main_title_suffix suffix of main title
#' @param plot=F whether output the ggplot
#' @param t_start = 30 ignore first 30 days' plot.
#' @param actions = c(rep(2, (t_end-t_begin+1) )), can bound with data, play as far as you like

#'
plotValidation <- function(data, newstate, l, t_begin = 50, t_end = 55, rep=1000, alpha=.1,
                MA_index = 0.4, nrow = 3, main_title_suffix = "", plot = T, t_start=1, poisson = T){

  Xl = data[[l]]
  xlab = as.Date(as.character(Xl$date), "%Y-%m-%d")
  name = names(data)
  population = Xl$population[1]
  Xl_shifted = shift_state(Xl, MA = MA_index, popu = population)
  XI = Xl_shifted$XI
  XR = Xl_shifted$XR
  XS = Xl_shifted$XS

  x_all_range = c(t_start:t_end)
  x_all_date = xlab[x_all_range]
  
  if(poisson){
    XI =  data.frame(x_date = x_all_date, count = Xl_shifted$XI[x_all_range])
    XR = data.frame(x_date = x_all_date, count = Xl_shifted$XR[x_all_range])
    XS = data.frame(x_date = x_all_date, count = Xl_shifted$XS[x_all_range])
    XNew = data.frame(x_date = x_all_date, count = newInfect(Xl_shifted$XR[x_all_range]))
    
    XIest = rbind(rep(Xl_shifted$XI[t_begin-t_start ], 3), newstate$XIest)
    XRest = rbind(rep(Xl_shifted$XR[t_begin-t_start ], 3), newstate$XRest)
    XSest = rbind(rep(Xl_shifted$XS[t_begin-t_start ], 3), newstate$XSest)
    XNest = rbind(rep(XNew[t_begin-t_start , 2], 3), newstate$XNest)
  }else{
    XI =  data.frame(x_date = x_all_date, count = Xl_shifted$XI[x_all_range]* population)
    XR = data.frame(x_date = x_all_date, count = Xl_shifted$XR[x_all_range]* population)
    XS = data.frame(x_date = x_all_date, count = Xl_shifted$XS[x_all_range]* population)
    XNew = data.frame(x_date = x_all_date, count = newInfect(Xl_shifted$XR[x_all_range]* population))
    
    XIest = rbind(rep(Xl_shifted$XI[t_begin-t_start ], 3)* population, newstate$XIest)
    XRest = rbind(rep(Xl_shifted$XR[t_begin-t_start ], 3)* population, newstate$XRest)
    XSest = rbind(rep(Xl_shifted$XS[t_begin-t_start ], 3)* population, newstate$XSest)
    XNest = rbind(rep(XNew[t_begin-t_start , 2], 3), newstate$XNest)
  }



  f1 = f1_func(data, l)
  f2 = f2_func(data, l)

  colours = c("black","red", "red","orangered")
  main_title = paste("Predict by first", as.character(t_begin - 1), "days data", main_title_suffix)

  x_pred_date = xlab[c((t_begin-t_start):t_end)]


  if (t_end > dim(Xl)[1]){
    x_pred_date = seq.Date(from=xlab[c((t_begin-t_start))],by="day", length.out = t_end-t_begin+t_start+1)
    x_pred_date = as.Date(as.character(x_pred_date), "%Y-%m-%d")
    #x_pred_date = as.POSIXct(x_pred_date,  origin = "1970-01-01")
    XI$x_date[c((t_begin-t_start): (t_end+t_start-1) )] = x_pred_date
  }
  XIest = data.frame(x_date = x_pred_date, count = XIest)
  XRest = data.frame(x_date = x_pred_date, count = XRest)
  XSest = data.frame(x_date = x_pred_date, count = XSest)
  XNest[XNest<0] = 0
  XNest = data.frame(x_date = x_pred_date, count = XNest)
  names(XIest) = c("x_date", "lower", "mean", "upper")
  names(XRest) <- c("x_date", "lower", "mean", "upper")
  names(XSest) = c("x_date", "lower", "mean", "upper")
  names(XNest) = c("x_date", "lower", "mean", "upper")

  if (t_end > dim(Xl)[1]){
    gg_I = ggplot(XIest, aes(x_date, count))+
      geom_line(data=XI, col = colours[1], na.rm = T) +
      geom_line(data = XIest,
                aes(x = x_date, y = lower),
                col = colours[3], linetype = 'dotted') +
      geom_line(data = XIest,
                aes(x = x_date, y = upper),
                col = colours[3], linetype = 'dotted') +
      geom_line(data = XIest,
                aes(x = x_date, y = mean),
                col = colours[4], linetype = 'solid') +
      ggtitle(paste("Latent Infectious in", name[l] ,sep = " ")) +
      xlab("Date") + ylab("Infectious (XI)")+
      theme(plot.title = element_text(hjust = 0.5))

    gg_R = ggplot(XRest, aes(x_date, count))+
      geom_line(data=XR, col = colours[1], na.rm = T) +
      geom_line(data = XRest,
                aes(x = x_date, y = lower),
                col = colours[3], linetype = 'dotted') +
      geom_line(data = XRest,
                aes(x = x_date, y = upper),
                col = colours[3], linetype = 'dotted') +
      geom_line(data = XRest,
                aes(x = x_date, y = mean),
                col = colours[4], linetype = 'solid') +
      ggtitle(paste("Cumulative Confirmed in", name[l] ,sep = " ")) +
      xlab("Date") + ylab("Infectious (XR)")+
      theme(plot.title = element_text(hjust = 0.5))

    gg_new = ggplot(XNest, aes(x_date, count))+
      geom_line(data=XNew, col = colours[1], na.rm = T) +
      geom_line(data = XNest,
                aes(x = x_date, y = lower),
                col = colours[3], linetype = 'dotted') +
      geom_line(data = XNest,
                aes(x = x_date, y = upper),
                col = colours[3], linetype = 'dotted') +
      geom_line(data = XNest,
                aes(x = x_date, y = mean),
                col = colours[4], linetype = 'solid') +
      ggtitle(paste("New Confirmed in", name[l] ,sep = " ")) +
      xlab("Date") + ylab("New Infectious (X_new)")+
      theme(plot.title = element_text(hjust = 0.5))

  } else{
    gg_I = ggplot(XI, aes(x_date, count))+
      geom_line(col = colours[1], na.rm = T) +
      geom_line(data = XIest,
                aes(x = x_date, y = lower),
                col = colours[3], linetype = 'dotted') +
      geom_line(data = XIest,
                aes(x = x_date, y = upper),
                col = colours[3], linetype = 'dotted') +
      geom_line(data = XIest,
                aes(x = x_date, y = mean),
                col = colours[4], linetype = 'solid') +
      ggtitle(paste("Latent Infectious in", name[l] ,sep = " ")) +
      xlab("Date") + ylab("Infectious (XI)")+
      theme(plot.title = element_text(hjust = 0.5))

    gg_R = ggplot(XR, aes(x_date, count))+
      geom_line(col = colours[1],na.rm = T) +
      geom_line(data = XRest,
                aes(x = x_date, y = lower),
                col = colours[3], linetype = 'dotted') +
      geom_line(data = XRest,
                aes(x = x_date, y = upper),
                col = colours[3], linetype = 'dotted') +
      geom_line(data = XRest,
                aes(x = x_date, y = mean),
                col = colours[4], linetype = 'solid') +
      ggtitle(paste("Cumulative Confirmed in", name[l] ,sep = " ")) +
      xlab("Date") + ylab("Infectious (XR)")+
      theme(plot.title = element_text(hjust = 0.5))

    gg_new = ggplot(XNew, aes(x_date, count))+
      geom_line(col = colours[1],na.rm = T) +
      geom_line(data = XNest,
                aes(x = x_date, y = lower),
                col = colours[3], linetype = 'dotted') +
      geom_line(data = XNest,
                aes(x = x_date, y = upper),
                col = colours[3], linetype = 'dotted') +
      geom_line(data = XNest,
                aes(x = x_date, y = mean),
                col = colours[4], linetype = 'solid') +
      ggtitle(paste("New Confirmed in", name[l] ,sep = " ")) +
      xlab("Date") + ylab("New Infectious (X_new)")+
      theme(plot.title = element_text(hjust = 0.5))

  }

  # geom_ribbon(data=predframe,aes(ymin=lwr,ymax=upr),alpha=0.3)
  if (plot) {
    grid.arrange(gg_I, gg_R,  gg_new, nrow = 3, top = textGrob(main_title, gp=gpar(fontsize=15,font=1)))
  }
  # return(result=list(gg_I, gg_R, gg_new))
}





