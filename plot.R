
# package -----------------------------------------------------------------

library("ggplot2")
library("reshape2")
library("ggpubr")

# data processing ---------------------------------------------------------

# rbind(c(
#   , 
#   ]
# c(


r1 = rbind(c(1.0600e+00, 1.0800e+00, 1.1000e+00, 1.4600e+00, 2.7851e+04, 1.0110e+01),                 
 c(9.1000e-01, 9.2000e-01, 1.0700e+00, 6.0000e-01, 6.2686e+04, 8.5900e+00),                                                                   
 c(2.6500e+00, 2.7500e+00, 3.2800e+00, 4.6600e+00, 1.3025e+05, 8.1400e+00),                                                                   
 c(6.2000e+00, 6.2600e+00, 6.6900e+00, 8.6300e+00, 1.3594e+05, 9.6800e+00))                                                                  


r2 = rbind(c(7.0000e-01, 5.7000e-01, 8.5000e-01, 1.4500e+00, 1.4136e+05, 1.0100e+01),                                                          
  c(5.3000e-01, 5.4000e-01, 8.7000e-01, 5.0000e-01, 1.0101e+05, 8.5900e+00),                                                               
  c(2.8200e+00, 2.9300e+00, 3.3000e+00, 4.6900e+00, 6.9906e+04, 8.1500e+00),                                                                   
  c(6.5000e+00, 6.5600e+00, 6.7300e+00, 8.6900e+00, 8.2243e+04, 9.6900e+00))                                                                   



r3 = rbind(c(9.2000e-01, 7.4000e-01, 7.5000e-01, 1.4700e+00, 6.5480e+05, 1.0120e+01),
  c(4.0000e-01, 3.7000e-01, 8.1000e-01, 4.4000e-01, 6.2550e+05, 8.6000e+00),
  c(2.9300e+00, 3.0500e+00, 3.2500e+00, 4.6500e+00, 2.8926e+05, 8.1600e+00),
  c(6.6100e+00, 6.6900e+00, 6.7100e+00, 8.6500e+00, 2.5924e+05, 9.7000e+00))

# data_settings = list(r1, r2, r3)
setting_range = c(4, 8, 12)

with_Naive_average = F
# trend = "sd_R"
trend = "day"

data_targets = list()
for (target in 1:4) {
  dat = rbind(r1[target, ], r2[target, ], r3[target, ]) # , r4[target, ]
  if (with_Naive_average) {
    dat = dat[,-5]
  }else{
    dat = dat[,-c(5,6)]
  }
  
  dat = cbind(setting_range, dat)
  dat = as.data.frame(dat)
  
  if (with_Naive_average) {
    names(dat) = c(trend, "DR","QV","IS", "DR without MARL",  "Naive Average") # "DR without Mean Field",
  }else{
    names(dat) = c(trend, "DR","QV","IS", "DR without MARL") # "DR without Mean Field",
  }
  
  if (trend == "sd_R") {
    mdf <- melt(dat, id.vars="sd_R", value.name="MSE", variable.name="method")
  }else{
    mdf <- melt(dat, id.vars="day", value.name="MSE", variable.name="method")
  }
  data_targets[[target]] = mdf
}





# plotting ----------------------------------------------------------------

threshold_levels = seq(80,110,10)

gg_one_target <-function(dat, threshold, x_axis = T, y_axis = T){
  n_est = length(levels(unique(data_targets[[l]][,2])))
  if (trend == "sd_R") {
    g = ggplot(data=dat, aes(x=sd_R, y=MSE, group = method, colour = method))
  }else{
    g = ggplot(data=dat, aes(x=day, y=MSE, group = method, colour = method))
  }
  
  
  g = g + geom_line() +
    geom_point( size = 3, aes(shape = method), fill="white")+ 
    scale_shape_manual(values=c(1:n_est)) +
    ggtitle(paste(sep = "", "Threshold = ", threshold)) + # paste(title_prefix, name[l] ,sep = " ")
    theme(plot.title = element_text(hjust = 0.5)) 
  # + 
    # scale_x_continuous(breaks = seq(7, 11, 1)) + 
    # scale_y_continuous(breaks = seq(0, 12, 1))
  if (x_axis) {
    if (trend == "sd_R") {
      g = g + xlab("sd_R")
    }else{
      g = g + xlab("Number of Days")
    }
  }
  if (y_axis) {
    y = g + ylab("MSE")
  }
  return(g)
}
# xlim

ggs = list()
for (l in c(1:4)) {
  ggs[[l]] = gg_one_target(data_targets[[l]], threshold_levels[l])
  # print(ggs[[l]])
}

g = ggarrange(ggs[[1]], ggs[[2]], ggs[[3]], ggs[[4]], ncol=2, nrow=2, common.legend = TRUE, legend="bottom")
print(g)

# https://rpkgs.datanovia.com/ggpubr/reference/ggarrange.html


# archive -----------------------------------------------------------------


# ggsave(g, height = ..., width = ...)


# dev.off()

# caption; legend; comparison; x and y axis
# one_plot <-function(){
#   
#   return()
#     
# }


  # annotate_figure(figure,
  #                 top = text_grob("Visualizing Tooth Growth", color = "red", face = "bold", size = 14),
  #                 bottom = text_grob("Data source: \n ToothGrowth data set", color = "blue",
  #                                    hjust = 1, x = 1, face = "italic", size = 10),
  #                 left = text_grob("Figure arranged using ggpubr", color = "green", rot = 90),
  #                 right = "I'm done, thanks :-)!",
  #                 fig.lab = "Figure 1", fig.lab.face = "bold"
  # )
# g = grid.arrange(gg_I, gg_R,  gg_new, nrow = 3, top = textGrob(main_title, gp=gpar(fontsize=15,font=1)))


# ggexport(g, filename = "/Users/mac/Desktop/test.png")

# , ncol = NULL,
#          nrow = NULL, width = 480, height = 480, pointsize = 12,
#          res = NA, verbose = TRUE)
# Off-policy evaluation results for the target policy with threshold X.

# 
# T1 = rbind(c(8.8000e-01, 6.8000e-01, 7.0000e-01, 1.5000e+00, 4.4516e+06, 1.0100e+01)
#            , c(3.0000e-01, 2.6000e-01, 7.6000e-01, 3.8000e-01, 3.6815e+05, 8.5900e+00)
#            , c(2.8900e+00, 3.0200e+00, 3.1700e+00, 4.6000e+00, 3.4414e+05, 8.1300e+00)
#            , c(6.6400e+00, 6.7200e+00, 6.6200e+00, 8.6200e+00, 7.8847e+05, 9.6800e+00))
# 
# 
# T2 = rbind(c(9.1000e-01, 7.2000e-01, 7.2000e-01, 1.4900e+00, 8.3047e+06, 1.0100e+01)
#            ,  c(3.4000e-01, 3.1000e-01, 7.6000e-01, 3.9000e-01, 3.1747e+05, 8.5900e+00)
#            ,  c(2.8800e+00, 3.0000e+00, 3.1900e+00, 4.6000e+00, 2.9499e+05, 8.1400e+00)
#            ,  c(6.6100e+00, 6.7000e+00, 6.6100e+00, 8.6200e+00, 7.7772e+05, 9.6800e+00))
# 
# 
# T3 = rbind(c(9.6000e-01, 7.9000e-01, 7.5000e-01, 1.5000e+00, 7.2443e+06, 1.0110e+01)
#            ,  c(4.5000e-01, 4.3000e-01, 7.9000e-01, 4.3000e-01, 3.2214e+05, 8.5900e+00)
#            ,  c(2.8800e+00, 3.0000e+00, 3.1800e+00, 4.6200e+00, 4.1990e+05, 8.1400e+00)
#            , c(6.6000e+00, 6.6900e+00, 6.6100e+00, 8.6300e+00, 6.8414e+05, 9.6800e+00))
# 
# 
# T4 = rbind(c(1.1500e+00, 1.0300e+00, 8.6000e-01, 1.5100e+00, 5.3854e+06, 1.0110e+01)
#            , c(7.4000e-01, 7.2000e-01, 8.8000e-01, 5.2000e-01, 4.4523e+05, 8.5900e+00)
#            , c(2.9200e+00, 3.0300e+00, 3.2100e+00, 4.6300e+00, 4.3148e+05, 8.1400e+00)
#            , c(6.6200e+00, 6.6900e+00, 6.6000e+00, 8.6500e+00, 7.2153e+05, 9.6800e+00))