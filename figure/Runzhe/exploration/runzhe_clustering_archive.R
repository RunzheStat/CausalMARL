
# DX = readRDS("/Users/mac/Google Drive/0CONFIDENT/Github/data/supplemental data/DXYArea_selected.rds")
# View(DX)

firstNoneZero <- function(dat){
  for (t in 1:16) {
    if(dat[t,3] != 0){
      return(t)
    }
  }
}

somePDFPath = "/Users/mac/Desktop/test.pdf"
pdf(file=somePDFPath)  

da = list()
betas  = matrix(rep(0,50 * 3) , nrow = 3, ncol = 50)
for (i in 1:50){
  city = data[[i]]
  TT = dim(city)[1] - 8
  tt = firstNoneZero(city)
  I_confirmed = city[tt:TT, 3] * 10^12
  I_confirmed = c(rep(0,8), I_confirmed)
  I = (data.table::shift(I_confirmed, n = 8, fill = NA, type="lead") - I_confirmed) 
  beta = (I + I_confirmed - data.table::shift(I + I_confirmed, n=1, fill=NA, type="lag")  ) / data.table::shift(I, n=1, fill=NA, type="lag")
  beta = beta[1:(length(beta)-8)]
  beta[is.na(beta)] = 0
  A = c(rep(0,8), city[tt:TT,5])
  A = A[1:(length(A)-8)]
  
  dat = cbind(beta, A)
  da[[i]] = dat
  co = cor(dat, use="pairwise.complete.obs", method="spearman")[2]
  
  cat(name[i], "\n")
  cat(co,"\n")
  for (j in 0:2) {
    a = dat[dat[,2] == j, 1]
    cat(mean( a, na.rm = T), sd( a, na.rm = T), "\n")
    betas[j + 1, i] = mean( a, na.rm = T)
  }
  
  plot(dat[,2],dat[,1])
  cat("\n\n")
}
betas = t(betas)

dev.off() 

# clustering_all ----------------------------------------------------------


somePDFPath = "/Users/mac/Desktop/clustering_name.pdf"
pdf(file=somePDFPath)  
# plot(t(betas[1:2,]))
betas[is.infinite(betas)] = 0

plot(betas[1:50,1] ,betas[1:50,2], type='n', ylim=c(0, 0.25))     
for (l in 1:50) {
  if(l <= 15){
    text(betas[l,1], betas[l,2], label=names(data)[l],col='red')
  }else{
    text(betas[l,1], betas[l,2], label=names(data)[l],col='blue')
  }
}


plot(betas[1:50,1], betas[1:50,3], type='n', ylim=c(0, 0.2))     
for (l in 1:50) {
  if(l <= 15){
    text(betas[l,1], betas[l,3], label=names(data)[l],col='red')
  }else{
    text(betas[l,1], betas[l,3], label=names(data)[l],col='blue')
  }
}


plot(betas[1:50,2], betas[1:50,3], type='n', ylim=c(0, 0.2))     
for (l in 1:50) {
  if(l <= 15){
    text(betas[l,2], betas[l,3], label=names(data)[l],col='red')
  }else{
    text(betas[l,2], betas[l,3], label=names(data)[l],col='blue')
  }
}

dev.off() 

# text(dat$x,dat$y2,label=1,col='green')
# text(dat$x,dat$y3,label=2,,col='red')
# 
# plot(1:50, betas[1,])
# plot(1:50, betas[2,])
# 
# 
# d = do.call(rbind, da)
# cor(d, use="pairwise.complete.obs", method="spearman")
# plot(d[,2],d[,1])




# clustering_NHB ----------------------------------------------------------

somePDFPath = "/Users/mac/Desktop/clustering_name_NHB.pdf"
pdf(file=somePDFPath)  
# betas = t(betas)
# plot(t(betas[1:2,]))
betas[is.infinite(betas)] = 0
betas[is.na(betas)] = 0
betas = data.frame(betas)
names(betas) = c("A1", "A2", "A3")

plot(betas[16:50,1] ,betas[16:50,2], type='n', ylim=c(0, max(betas[16:50,2])), xlim=c(0, max(betas[16:50,1])) )     
for (l in 16:50) {
  text(betas[l,1], betas[l,2], label=names(data)[l],col='blue')
}


plot(betas[16:50,1] ,betas[16:50,3], type='n', ylim=c(0, max(betas[16:50,3])), xlim=c(0, max(betas[16:50,1])) )     
for (l in 16:50) {
  text(betas[l,1], betas[l,3], label=names(data)[l],col='blue')
}



plot(betas[16:50,2] ,betas[16:50,3], type='n', ylim=c(0, max(betas[16:50,3])), xlim=c(0, max(betas[16:50,2])) )     
for (l in 16:50) {
  text(betas[l,2], betas[l,3], label=names(data)[l],col='blue')
}


dev.off() 

