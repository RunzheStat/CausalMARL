rm(list=ls())
library(xzhang97wuhan)
library(invgamma)
library(data.table)
library(ggplot2)
library(gridExtra)
library(grid)
library(crmPack)
library("invgamma")

#### debug of unstable numerical result ####
data_cn = readRDS("data/covid_cn.rds")
Xl = data_cn[[37]]
getSIRMAP(data_cn, t0=34,lag=8, J=3,region_index=c(37), echo=T)
getSIRMAP(data_cn, t0=35,lag=8, J=3,region_index=c(37), echo=T)
shift_state(data_cn[[37]])
data = data_cn
getbetaforOneCity(data_cn, l=37, t0=42, MA = 0.4, lag = 8, t_start = 0)

#### prediction for china ####
data_cn = readRDS("data/covid_cn.rds")
params_cn = getSIRMAP(data_cn, t0=42,lag=8, J=3,region_index=c(1:50))
round(params_cn,5)
theta_cn = params_cn[1:8,1]
data=data_cn
theta=theta_cn
l=1
t_begin = 43
t_end = 100
rep = 1000
alpha=.1
MA_index = 0.4
lag = 8


action=1
actions =c(rep(action, (t_end-t_begin+1) ))
oldstate = prediction(data, theta, 1, t_begin = t_begin, t_end = t_end, rep=rep, alpha = alpha, actions = actions, MA_index = 0.4)
l=1
for(l in 1:50){
  newstate = prediction(data, theta, 2, t_begin = t_begin, t_end = t_end, rep=rep, alpha = alpha, actions = actions, MA_index = 0.4)

}


#
# filename = paste(sep="", "figure/china_future_action_",action-1,".pdf")
# pdf(filename)
#
# actions =c(rep(action, (t_end-t_begin+1) ))
#
# for(l in 1:50){
#   newstate = prediction(data, theta, l, t_begin = t_begin, t_end = t_end, rep=rep, alpha = alpha, actions = actions, MA_index = 0.4)
#   plotValidation(data, newstate, l, t_begin = t_begin, t_end = t_end)
# }
#
# dev.off()

#### train theta from america
data_us = readRDS("data/covid_us.rds")
data_us =  mergeUSdata(data_us)
data_us[[58]]$suspect
names(data_us)[58] = "US"
length(data_us)
dim(data_us[[1]])
as.Date(55, origin=range(data_us[[1]]$date)[1])
# 58 the whole america
params_all_seperate = getSIRMAP(data_us, t0=55,lag=8, J=3, region_index=c(1:57), echo=F)
params_all_seperate
params_all_us = getSIRMAP(data_us, t0=55,lag=8, J=3, region_index=c(58), echo=F)
params_all_us
theta_us = params_all_seperate[1:8,1]
data = data_us
theta = theta_us
l=1
t_begin = 56
as.Date(t_begin-1, origin=range(data_us[[1]]$date)[1])
t_end = 200
as.Date(t_end-1, origin=range(data_us[[1]]$date)[1])
rep = 1000
alpha=.1
action=3

#
filename = paste(sep="", "figure/US_future_action_",action-1,".pdf")
#pdf(filename)
actions =c(rep(action, (t_end-t_begin+1) ))

#'Only for testing merge result purpose.
#'
#'mat0 = matrix(0, nrow=(t_end-t_begin+1), ncol = 3)
#'newstate = list( mat0,mat0,mat0,mat0 )
#'newstate[[i]] = oldstate[[i]]+newstate[[i]]
#'names(newstate) = names(oldstate)

for(l in 1:58){
  newstate = prediction(data, theta, l, t_begin = t_begin, t_end = t_end, rep=rep, alpha = alpha, actions = actions, MA_index = 0.4)
  plotValidation(data, newstate, l, t_begin = t_begin, t_end = t_end)
}

dev.off()
