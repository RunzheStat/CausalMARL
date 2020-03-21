# Environment for Runzhe --------------------------------------------------

# # data = readRDS("data/covid.rds")
data_cn= readRDS("/Users/mac/Google Drive/Confident/data/covid_cn_count.rds")
library(rstanarm)
library(ggplot2)
library(gridExtra)
library(grid)
install.load::install_load("invgamma")

main_path = "/Users/mac/Google Drive/Confident/code/code in package/"
# main_path = "/Users/mac/Desktop/code/code in package/"

source(paste(main_path,"utility.R",sep = ""))
source(paste(main_path,"SIR_model.R",sep = ""))
source(paste(main_path,"SIR_predict.R",sep = ""))
              



for (region_index in list(c(1:50), c(1:15), c(16:50))) {
  cat(region_index,"\n\n")
  for (t0 in c(20,25,30,35,40)) {
    a = getSIRMAP(data=data_cn, t0=t0, lag=8, J=3, u_2 = 0.5, u_3 = 0.1, region_index=region_index, 
              MA_index = NA, echo = F, t_start = 0)
    cat(a,"\n")
  }
}

getSIRMAP(data=data_cn, t0=42, lag=8, J=3, u_2 = 0.5, u_3 = 0.1, region_index=region_index, 
                  MA_index = NA, echo = F, t_start = 0)

for(l in 1:50){
  newstate = prediction(data = data_cn, theta = theta[1:4], l = l, t_begin = 20, 
                        t_end = 42, rep=1000, alpha = 0.95, actions = NA, MA_index = NA)
  plotValidation(data_cn, newstate, l, t_begin = 20, t_end = t_end)
}
# prediction(data = , theta = theta[1:4], l = 1, t_begin = 31, t_end = 42, rep=1000, alpha = 0.9, 
#            actions = NA, MA_index = NA, echo=FALSE, poisson = T)
# getSIRMAP(data=data_cn, t0=40, lag=8, J=3, u_2 = 0.5, u_3 = 0.1, region_index=c(16:50), 
#           MA_index = NA, echo = F, t_start = 0)

rep=1000
lag=8


plot_pred_China <- function(t0 = 25, t_end=42, alpha=.02, range = "NHB"){
  t_begin=t0+1
  
  filename = paste(sep="", "figure/",range,"_", t0,"_", t_end,"_", alpha * 100, ".pdf")
  pdf(filename)
  if (range == "NHB")region_index=c(16:50)
  if (range == "HB")region_index=c(1:15)
  if (range == "all")region_index=c(1:50)
  params_shift = getSIRMAP(data_cn, t0 = t0, lag = 8, J = 3, region_index = region_index)
  theta = params_shift[1:4]
  
  for (l in region_index) {
    newstate = prediction(data=data_cn, theta=theta, l=l, t_begin = t_begin, t_end=t_end, rep=rep, alpha = alpha)
    plotValidation(data_cn, newstate, l, t_begin = t_begin, t_end = t_end, rep=rep, alpha=alpha)
    cat(l,"\n")
  }
  dev.off()
}

library(parallel)
for (t_end in c(42,50)) {
for (range in list("NHB","HB","all")) {
  for (alpha in c(0.01,0.02,0.05)) {
      once_t0 <-function(t0){
        plot_pred_China(t0 = t0, t_end=t_end, alpha=alpha, range = range)
      }
      mclapply(c(20,25,30,35), once_t0, mc.cores = 4)
  }
}
  cat(t0,t_end, alpha,range, "DONE!")
}
