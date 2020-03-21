

prediction <- function(data, theta, l, t_begin = 31, t_end = 42, rep=1000, alpha = 0.4, actions = NA, MA = 0.4){
  ####### Based on estimated parameter until t_begin, construct PI until t_end, following observed actions or specified actions
  ### Start from S_{t0}, using A_{t0}, ..., predict S_{t0+1}, .... 
  ## Args
  #   t_begin: = t0 + 1
  #   actions: if "NA", then use observed  actions to do validation; otherwise, use specified actions to evaluate policies
  #
  ## Output: the estimated states for (t_begin, t_end]

  ## preprocess data
  f1 = f1_func(data, l)
  f2 = f2_func(data, l)
  Xl = data[[l]]
  tl = Xl$tl[1]
  population = Xl[1,10]

  Xl_shifted = shift_state(Xl, MA = MA)
  XI = Xl_shifted[[1]]
  XR = Xl_shifted[[2]]
  XS = Xl_shifted[[3]]
  A =  Xl_shifted[[4]]
  X_lts = cbind(XS, XI, XR)

  ## Initialization
  XIest = matrix(0,nrow = c(t_end-t_begin+1), ncol=3)
  rownames(XIest) <- c(t_begin:t_end)
  XNest = XSest = XRest = XIest
  XNtraj = XStraj = XRtraj = XItraj = matrix(0,nrow = rep, ncol=c(t_end-t_begin+1))

  # Generate trajectories
  for (r in 1:rep){
    for (t in t_begin:t_end){
      
      if(!is.na(actions)){
        # cat(X_lts[t-1,], actions[t - t_begin + 1], "\n")
        X_lts[t,] = f(X_lts[t-1,], actions[t - t_begin + 1], theta)
      }else{
        X_lts[t,] = f(X_lts[t-1,], A[t-1], theta)
      }
      
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
  
  
  newstate = list(XSest=XSest * population, XIest = XIest * population, XRest = XRest * population, XNest = XNest * population)
  return(newstate)
}


plotPrediction <- function(newstate, l, t_begin = 31, t_end = 42, 
                           MA = 0.4, nrow = 3, main_title_suffix = ""){
  #
  
  Xl = data[[l]]
  xlab = as.Date(as.character(Xl$dates), "%Y-%m-%d");head(xlab)
  name = names(data)
  population = Xl[1,10]
  Xl_shifted = shift_state(Xl, MA = MA)
  x_all_range = c(1:t_end)
  x_all_date = xlab[x_all_range]
  
  XI =  data.frame(x_date = x_all_date, count = Xl_shifted[[1]][x_all_range]* population)
  XR = data.frame(x_date = x_all_date, count = Xl_shifted[[2]][x_all_range]* population)
  XS = data.frame(x_date = x_all_date, count = Xl_shifted[[3]][x_all_range]* population)
  XNew = data.frame(x_date = x_all_date, count = newInfect(Xl_shifted[[2]][x_all_range]* population))
  
  XIest = rbind(rep(Xl_shifted[[1]][t_begin - 1], 3)* population, newstate$XIest)
  XRest = rbind(rep(Xl_shifted[[2]][t_begin - 1], 3)* population, newstate$XRest)
  XSest = rbind(rep(Xl_shifted[[3]][t_begin - 1], 3)* population, newstate$XSest)
  XNest = rbind(rep(XNew[t_begin - 1, 2], 3), newstate$XNest)
  
  f1 = f1_func(data, l)
  f2 = f2_func(data, l)
  
  colours = c("black","red", "red","orangered")
  main_title = paste("Predict by first", as.character(t_begin - 1), "days data", main_title_suffix)
  
  t_begin = t_begin - 1
  x_pred_date = xlab[c((t_begin):t_end)]
  
  XIest = data.frame(x_date = x_pred_date, count = XIest)
  XRest = data.frame(x_date = x_pred_date, count = XRest)
  XSest = data.frame(x_date = x_pred_date, count = XSest)
  XNest[XNest<0] = 0
  XNest = data.frame(x_date = x_pred_date, count = XNest)
  names(XIest) = c("x_date", "lower", "mean", "upper")
  names(XRest) <- c("x_date", "lower", "mean", "upper")
  names(XSest) = c("x_date", "lower", "mean", "upper")
  names(XNest) = c("x_date", "lower", "mean", "upper")
  
  gg_I = ggplot(XI, aes(x_date, count))+
    geom_line(col = colours[1],) +  
    geom_line(data = XIest, 
              aes(x = x_date, y = lower), 
              col = colours[3], linetype = 'dotted') +
    geom_line(data = XIest, 
              aes(x = x_date, y = upper), 
              col = colours[3], linetype = 'dotted') +
    geom_line(data = XIest, 
              aes(x = x_date, y = mean), 
              col = colours[4], linetype = 'solid') +
    ggtitle(paste("Prediction of Latent Infectious (XI) in", name[l] ,sep = " ")) +
    xlab("Date") + ylab("Infectious (XI)")+
    theme(plot.title = element_text(hjust = 0.5))
  
  gg_R = ggplot(XR, aes(x_date, count))+
    geom_line(col = colours[1],) +  
    geom_line(data = XRest, 
              aes(x = x_date, y = lower), 
              col = colours[3], linetype = 'dotted') +
    geom_line(data = XRest, 
              aes(x = x_date, y = upper), 
              col = colours[3], linetype = 'dotted') +
    geom_line(data = XRest, 
              aes(x = x_date, y = mean), 
              col = colours[4], linetype = 'solid') +
    ggtitle(paste("Prediction of Cumulative Confirmed (XR) in", name[l] ,sep = " ")) +
    xlab("Date") + ylab("Infectious (XR)")+
    theme(plot.title = element_text(hjust = 0.5))

  gg_new = ggplot(XNew, aes(x_date, count))+
    geom_line(col = colours[1],) +
    geom_line(data = XNest,
              aes(x = x_date, y = lower),
              col = colours[3], linetype = 'dotted') +
    geom_line(data = XNest,
              aes(x = x_date, y = upper),
              col = colours[3], linetype = 'dotted') +
    geom_line(data = XNest,
              aes(x = x_date, y = mean),
              col = colours[4], linetype = 'solid') +
    ggtitle(paste("Prediction of New Confirmed Cases in", name[l] ,sep = " ")) +
    xlab("Date") + ylab("New Infectious (X_new)")+
    theme(plot.title = element_text(hjust = 0.5))
  
  # geom_ribbon(data=predframe,aes(ymin=lwr,ymax=upr),alpha=0.3)
  # Xinyu does not require this. Can just use her plotting or rely on er.
  grid.arrange(gg_I, gg_R,  gg_new, nrow = nrow, top = textGrob(main_title, gp=gpar(fontsize=15,font=1)))
  
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


