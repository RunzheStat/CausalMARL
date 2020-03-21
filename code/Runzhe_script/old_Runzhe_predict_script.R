
# Parameters --------------------------------------------------------------


t0 = 34
alpha = 0.05
t_begin = t0 + 1


getSIRMAP(data=data_cn, t0=25, lag=8, J=3, u_2 = 0.5, u_3 = 0.1, region_index=c(1:50), MA_index = NA, echo = F, t_start = 0)

getSIRPosteriorForEachDay(t0 = t0, lag = 8, J = 3, city_index = c(1:15))
getSIRPosteriorForEachDay(t0 = t0, lag = 8, J = 3, city_index = c(1:50))
getSIRPosteriorForEachDay(t0 = t0, lag = 8, J = 3, city_index = c(16:50))


getSIRPosteriorForEachDay(t0 = t0, lag = 8, J = 3, city_index = c(1:50))


getSIRPosteriorForEachDay(t0 = 35, lag = 8, J = 3, city_index = c(37), echo = T)

# Evaluation --------------------------------------------------------------

t0 = 34
alpha = 0.05
params = getSIRPosteriorForEachDay(t0 = 34, lag = 8, J = 3, city_index = c(1:50))
theta = params[1:8,1]
t_begin = 20
path = paste("/Users/mac/Desktop/evaluation", as.character((1-alpha*2)*100),"_", as.character(t_begin),".pdf", sep = "" )
pdf(path)

for (l in 10:20) {
  for (i in 1:3) {
    newstate = prediction(data, theta, l = l, t_begin = t_begin, t_end = 42, rep=1000, alpha = alpha, actions = rep(i, 55))
    plotPrediction(newstate, l = l, t_begin = t_begin, main_title_suffix = as.character(i), plot = T)
  }
  cat(l,"done! ")
}
dev.off()


# HB-------------------------------------------------------------------------


t0 = 25
alpha = 0.05
t_begin = t0 + 1


params = getSIRPosteriorForEachDay(t0 = t0, lag = 8, J = 3, city_index = c(1:15))
theta = params[1:8,1]
path = paste("/Users/mac/Desktop/HB_", as.character((1-alpha)*100),"_", as.character(t0),".pdf", sep = "" )
pdf(path)
for (l in 1:15) {
  newstate = prediction(data, theta, l, t_begin = t_begin, t_end = 42, rep=1000, alpha = alpha)
  plotPrediction(newstate, l, t_begin = t_begin, plot = T)
}
dev.off()




# NHB ---------------------------------------------------------------------
index_NHB = c(16:50)
# index_outlier = c(31)
# index_NHB = index_NHB[!index_NHB %in% index_outlier]
params = getSIRPosteriorForEachDay(t0 = t0, lag = 8, J = 3, city_index = index_NHB, echo =  F)
theta = params[1:8,1]
path = paste("/Users/mac/Desktop/NHB_", as.character((1-alpha)*100),"_", as.character(t0),".pdf", sep = "" )
pdf(path)
for (l in index_NHB) {
  newstate = prediction(data, theta, l, t_begin = t_begin, t_end = 42, rep=1000, alpha = alpha)
  plotPrediction(newstate, l, t_begin = t_begin, plot = T)
}
dev.off()


# All ---------------------------------------------------------------------

t0 = 20
alpha = 0.05
t_begin = t0 + 1

index_NHB = c(1:50)
# index_outlier = c(31)
# index_NHB = index_NHB[!index_NHB %in% index_outlier]
params = getSIRPosteriorForEachDay(t0 = t0, lag = 8, J = 3, city_index = index_NHB, echo =  T)
theta = params[1:8,1]
path = paste("/Users/mac/Desktop/all_", as.character((1-alpha*2)*100),"_", as.character(t0),".pdf", sep = "" )
pdf(path)
for (l in index_NHB) {
  newstate = prediction(data, theta, l, t_begin = t_begin, t_end = 42, rep=1000, alpha = alpha)
  plotPrediction(newstate, l, t_begin = t_begin, plot = T)
}
dev.off()


# for (l in 1:15) {
#   data[[l]][,5][data[[l]][,5] == 0] = 2
# }

# Try cross-sectional validation ------------------------------------------
# a[!(a %in% b)]


t0 = 34
alpha = 0.05
t_begin = 10


params = getSIRPosteriorForEachDay(t0 = t0, lag = 8, J = 3, city_index = c(1:10))

theta = params[1:8,1]
pdf("/Users/mac/Desktop/HB_cross_90.pdf")
for (l in 11:15) {
  newstate = prediction(data, theta, l, t_begin = t_begin, t_end = 42, rep=1000, alpha = alpha)
  plotPrediction(newstate, l, t_begin = t_begin, plot = T)
}
dev.off()

params = getSIRPosteriorForEachDay(t0 = t0, lag = 8, J = 3, city_index = c(16:33))

theta = params[1:8,1]
pdf("/Users/mac/Desktop/NHB_cross_90.pdf")
for (l in 34:50) {
  newstate = prediction(data, theta, l, t_begin = t_begin, t_end = 42, rep=1000, alpha = alpha)
  plotPrediction(newstate, l, t_begin = t_begin, plot = T)
}
dev.off()

## Two-fold
pdf("/Users/mac/Desktop/all_cross_95_10.pdf")
params = getSIRPosteriorForEachDay(t0 = t0, lag = 8, J = 3, city_index = c(26:50))
theta = params[1:8,1]
for (l in 1:25) {
  newstate = prediction(data, theta, l, t_begin = t_begin, t_end = 42, rep=1000, alpha = alpha)
  plotPrediction(newstate, l, t_begin = t_begin, plot = T)
}

params = getSIRPosteriorForEachDay(t0 = t0, lag = 8, J = 3, city_index = c(1:25))
theta = params[1:8,1]
for (l in 26:50) {
  newstate = prediction(data, theta, l, t_begin = t_begin, t_end = 42, rep=1000, alpha = alpha)
  plotPrediction(newstate, l, t_begin = t_begin, plot = T)
}
dev.off()

