

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
      if(is.na(actions)){
        X_lts[t,] = f(X_lts[t-1,], A[t-1], theta)
      }else{
        X_lts[t,] = f(X_lts[t-1,], actions[t - t_begin + 1], theta)
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
                           MA = 0.4){
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
  main_title = paste("Predict by first", as.character(t_begin - 1), "days data")

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
    ggtitle(paste("Latent Infectious in", name[l] ,sep = " ")) +
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
    ggtitle(paste("Cumulative Confirmed in", name[l] ,sep = " ")) +
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
    ggtitle(paste("New Confirmed in", name[l] ,sep = " ")) +
    xlab("Date") + ylab("New Infectious (X_new)")+
    theme(plot.title = element_text(hjust = 0.5))

  # geom_ribbon(data=predframe,aes(ymin=lwr,ymax=upr),alpha=0.3)
  # grid.arrange(gg_I, gg_R,  gg_new, nrow = 3, top = textGrob(main_title, gp=gpar(fontsize=15,font=1)))
  return(result=list(gg_I, gg_R, gg_new))
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




plotPrediction_origin <- function(XSest, XIest, XRest, l, t_begin = 31, t_end = 42){
  f1 = f1_func(data, l)
  f2 = f2_func(data, l)
  Xl = data[[l]]
  population = Xl[1,10]
  tl = Xl$tl[1]

  Xl_shifted = shift_state(Xl)
  XI = Xl_shifted[[1]] * population
  XR = Xl_shifted[[2]] * population
  XS = Xl_shifted[[3]] * population
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

arrangeGgplot <- function(ggplots, n, p){
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
  } else{
    for (i in 1:floor(n/p)){
      grid.newpage()
      pushViewport(viewport(layout = grid.layout(3,2)))
      print(ggplots[[p*i-(p-1)]], vp = viewport(layout.pos.row = 1, layout.pos.col = 1))
      print(ggplots[[p*i-(p-2)]], vp = viewport(layout.pos.row = 2, layout.pos.col = 1))
      print(ggplots[[p*i-(p-3)]], vp = viewport(layout.pos.row = 3, layout.pos.col = 1))
      print(ggplots[[p*i-(p-4)]], vp = viewport(layout.pos.row = 1, layout.pos.col = 2))
      print(ggplots[[p*i-(p-5)]], vp = viewport(layout.pos.row = 2, layout.pos.col = 2))
      print(ggplots[[p*i-(p-6)]], vp = viewport(layout.pos.row = 3, layout.pos.col = 2))
    }
    i = i + 1
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(3,2)))
    print(ggplots[[p*i-(p-1)]], vp = viewport(layout.pos.row = 1, layout.pos.col = 1))
    print(ggplots[[p*i-(p-2)]], vp = viewport(layout.pos.row = 2, layout.pos.col = 1))
    print(ggplots[[p*i-(p-3)]], vp = viewport(layout.pos.row = 3, layout.pos.col = 1))
  }

}
