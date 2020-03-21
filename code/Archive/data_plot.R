plotWuhanQianxi <- function(city="wuhan", index=c(-22:43), legend.xaxis=as.Date("2019-01-30")){
  qianxi <- read.csv("data/qianxi.csv", header = T)[-1,]
  city_index = which(names(qianxi) == city)
  colours = c("black","turquoise3", "darkgoldenrod1","orangered")
  date = qianxi$date[index+23]
  xlab = as.Date(as.character(date), "%Y%m%d")
  qianxi2020 <- as.numeric( as.character(qianxi[index+99, city_index] ))
  qianxi2019 <- as.numeric( as.character(qianxi[index+99-76, city_index]))
  ylim = max(range(qianxi2019), range(qianxi2020))*1.8
  plot(xlab, qianxi2019, type="l", ylim=c(0,ylim), ylab="qianxi_index",
       main=paste(sep="",city,"'s qianxi"), xlab="date", col=colours[2])
  lines(xlab, qianxi2020, type="l", col=colours[3])
  r=qianxi2019 / qianxi2020
  lines(xlab, r, col=colours[4])
  legend("top",c("m19","m20","r=m19/m20"),cex=.7 ,lty=1, col=colours[c(2:4)])
}

plotQQianxiStrengthPdf <- function(){
  qianxi <- read.csv("data/qianxi.csv", header = T)[,-1]
  cities = names(qianxi)[-c(1:2)]
  pdf("data/qianxi.pdf")
  par(mfrow=c(2,2))
  for(city in cities){
    plotWuhanQianxi(city)
  }
  dev.off()
}

dataCleanQianxiRatio <- function(){
  qianxi <- read.csv("data/qianxi.csv", header = T)[,-1]
  qianxi2020 <- qianxi[100:143, ]
  qianxi2019 <- qianxi[24:67, ]
  ratio <- qianxi2019/qianxi2020
  qianxi_ratio <- ratio[,-1]
  qianxi_ratio[,1] = qianxi$code[24:67]
  date2020 = qianxi$date[100:143]
  date2019 = qianxi$date[24:67]
  qianxi_ratio = cbind(date2019, date2020, qianxi_ratio)
  return(qianxi_ratio)
}

plotQianxiRatioAll <- function(){
  pdf("data/qianxi_ratio.pdf")
  par(mfrow=c(3,3))
  colours = c("black","turquoise3", "darkgoldenrod1","orangered")
  for(i in c(4:53)){
    plot(qianxi_ratio$date, qianxi_ratio[,i], type="l",  main=paste(sep="",names(qianxi_ratio)[i],"'s ratio"), ylim=c(0,8) ,xlab="date", col=colours[4])
  }
  dev.off()
}


plotQianxiwithAction <- function(l=1, index=c(1:42), legend.xaxis=as.Date("2019-02-10")){
  city_action = action[,2+l]
  city = names(action)[2+l]

  colours = c("black","turquoise3", "darkgoldenrod1","orangered")
  date = qianxi$date[index+23]
  xlab = as.Date(as.character(date), "%Y%m%d")
  qianxi2020 <- as.numeric( as.character(qianxi[index+99, 2+l] ))
  qianxi2019 <- as.numeric( as.character(qianxi[index+99-76, 2+l]))
  ylim = max(range(qianxi2019), range(qianxi2020))*1.8
  plot(xlab, qianxi2019, type="l", ylim=c(-1,ylim),
       main=paste(sep="", "city ", l,": ", city,""), xlab="date", col=colours[2])
  lines(xlab, qianxi2020, type="l", col=colours[3])
  r=qianxi2019/qianxi2020
  lines(xlab, scale(r)+2, col=colours[4])
  lines(xlab, city_action[index] )
  legend("topleft",c("m19","m20","r=m19/m20","action"),cex=.5 ,lty=1, col=colours[c(2:4,1)])
}

plotQQianxiWithActionPdf <- function(){
  action  = read.csv("data/action.csv", skip=4)
  qianxi <- read.csv("data/qianxi.csv", header = T)[-1,-1]
  pdf("figure/qianxi_action_all.pdf")
  par(mfrow=c(3,3))
  for(l in 1:50){
    plotQianxiwithAction(l)
  }
  dev.off()
}
