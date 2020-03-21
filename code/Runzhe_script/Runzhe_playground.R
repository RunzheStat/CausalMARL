library(data.table)
data = readRDS("/Users/mac/Google Drive/Confident/data/covid.rds")
name = names(data)

View(data[[20]])


# Visualization: cumulative infected number / actionsfor each city ------------------------------------------------
# 
# pdf("/Users/mac/Desktop/XR_all.pdf")
# for (i in 1:50) {
#   XR_all = data.frame(count = data[[i]][,3]*data[[i]][1,10], date = c(1:length(data[[i]][,3])))
#   XR_one = ggplot(data= XR_all, aes(x=date, y=count)) +
#     geom_line()+
#     geom_point() +
#     ggtitle(name[i]) +
#     xlab(i) 
#   grid.arrange(XR_one, nrow = 1)
# }
# dev.off()

for (l in 1:50) {
  city  = data[[l]]
  cat(city$action,"\n")
}


# Archive -----------------------------------------------------------------




# for (city in data){
#   TT = dim(city)[1]
#   tt = firstNoneZero(city)
#   I = city[tt:TT, 3] * 10^8
#   beta = (I - shift(I, n=1, fill=NA, type="lag")) / shift(I, n=1, fill=NA, type="lag")
#   # co = cor.test(beta, city[,5],  method = "spearman")
#   dat = cbind(beta, shift(city[tt:TT,5], n=8, fill=NA, type="lag") )
#   dat[is.na(dat)] <- 0
#   # plot(city[,5], beta)
#   da[[i]] = dat
#   co = cor(dat, use="pairwise.complete.obs", method="spearman")
#   i = i + 1
#   # cat(co,"\n")
#   for (i in 0:2) {
#     a = dat[dat[,2] == i, 1]
#     # cat(mean( a[a!=0], na.rm = T), sd( a[a!=0], na.rm = T), "\n")
#     cat(mean( a, na.rm = T), sd( a, na.rm = T), "\n")
#   }
#   cat("\n\n")
# }