rm(list=ls())
library(xzhang97wuhan)
library(invgamma)
library(data.table)
library(ggplot2)
library(gridExtra)
library(grid)
library(crmPack)
library("invgamma")

#### plot validation for china  ####
data = readRDS("data/covid_cn.rds")
t0 = 30
# params_noshift = getSIRMAP(data, t0 = t0, lag = 0, J = 3, region_index = c(1:50))
# for (region_index in list(c(1:50), c(1:15), c(16:50))) {
#   cat(region_index,"\n\n")
#   for (t0 in c(20,25,30,35,40)) {
#     a = getSIRMAP(data=data_cn, t0=t0, lag=8, J=3, u_2 = 0.5, u_3 = 0.1, region_index=region_index, 
#                   MA_index = NA, echo = F, t_start = 0)
#     cat(a,"\n")
#   }
# }
params_shift = getSIRMAP(data, t0 = t0, lag = 8, J = 3, region_index = c(1:50))
round(params_noshift,6)
round(params_shift,6)
theta = params_shift[1:8,1]
# use t0-1 for training

t0 = 25
l=1
t_begin=t0+1
t_end=42
rep=1000
alpha=.02
Xl = data[[l]]
lag=8
newstate = prediction(data=data, theta=theta, l=l, t_begin = t_begin, t_end=t_end, rep=rep, alpha = alpha)
filename = paste(sep="", "figure/china_",t0,".pdf")
pdf(filename)

params_shift = getSIRMAP(data, t0 = t0, lag = 8, J = 3, region_index = c(16:50))
theta = params_shift[1:4]

for (l in 16:50) {
  newstate = prediction(data=data, theta=theta, l=l, t_begin = t_begin, t_end=t_end, rep=rep, alpha = alpha)
  plotValidation(data, newstate, l, t_begin = t_begin, t_end = t_end, rep=rep, alpha=alpha)
  cat(l,"\n")
}
dev.off()



#### plot validation for us ####
data_us = readRDS("data/covid_us.rds")

length(data_us)
str( data_us[[1]] )
WV = addTltoData(data_us[[57]])
WV$tl
for (l in 1:57){
  tl = addTltoData(data_us[[l]])$tl[1]
  cat("l=,",l,"tl=",tl,"\n")
}
serious_state_index <- c(2, 3, 4, 10, 35)
names(data_us)[serious_state_index]
data=data_us
t0=40
l=37
t_start=1
lag=8

getbetaforOneCity(data_us, l, t0, lag = 8, MA_index = 0.4, t_start = 1)
getSIRMAP(data_us, t0=55,lag=8, J=3,region_index=serious_state_index, echo=F)
getSIRMAP(data_us, t0=55,lag=8, J=3,region_index=c(1:56)[-serious_state_index], echo=F)
params_all = getSIRMAP(data, t0 = t0, region_index = c(1:56), echo=T) #t0=39:54
params_all
theta_us = params_all[1:8,1]
theta_us
filename = paste(sep="", "figure/US_",t0,".pdf")
pdf(filename)
t_end=55
rep=1000
alpha=.1
for (l in 1:57) {
  newstate = prediction(data=data_us, theta=theta_us, l=l, t_begin = t0+1, t_end=t_end, rep=rep, alpha = alpha)
  plotValidation(data_us, newstate, l, t_begin = t0+1, t_end = t_end, rep=rep, alpha=alpha)
}

dev.off()



