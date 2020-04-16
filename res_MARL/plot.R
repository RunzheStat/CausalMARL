library("ggplot2")
library("reshape2")
library("ggpubr")
library(latex2exp)
library(readr)

# Data --------------------------------------------------------------------
res = list()

y_var = "MSE"
# y_var = "bias"
# y_var = "std"

trend = "sd_R"
# trend = "day"
# trend = "Time"


path_plot = "/Users/mac/Desktop/"

for (i in 1:n_axis) {# x-axis
  path = paste(sep = "", path_plot, "res_sd_", y_var, i, ".txt")
  res_1 <- read_csv(path,
                    col_names = FALSE)
  # res_1 = res_1[c(3,4,6,7),]
  res[[i]] = abs(res_1)
}

# Targets / Axis-----------------------------------------------------------------

target_num = 4
threshold_levels = c(100, 105, 110, 115)

n_axis = 3
# setting_range = seq(3,9,2)

# Competing ----------------------------------------------------------------

with_Naive_average = T
with_QV = F

# Setting-fixed -----------------------------------------------------------

data_targets = list()

for (target in 1:target_num) {
  da = list()
  for (i in c(1:length(setting_range))) {
    da[[i]] = res[[i]][target, ]
  }
  
  dat = do.call(rbind, da)
  if (with_Naive_average) {
    dat = dat[,-5]
  }else{
    dat = dat[,-c(5,6)]
  }
  
  if (!with_QV) {
    dat = dat[,-2]
  }
  
  dat = cbind(setting_range, dat)
  dat = as.data.frame(dat)
  
  if (with_Naive_average) {
    if (with_QV) {
      names(dat) = c(trend, "DR","QV","IS", "DR-NS",  "Naive Average")
    }else{
      names(dat) = c(trend, "DR", "IS", "DR-NS",  "Naive Average")
    }
  }else{
    if (with_QV) {
      names(dat) = c(trend, "DR","QV","IS", "DR-NS") # "DR without Mean Field",
    }else{
      names(dat) = c(trend, "DR","IS", "DR-NS") # "DR without Mean Field",
    }
  }
  
  if (trend == "sd_R") {
    mdf <- melt(dat, id.vars="sd_R", value.name="MSE", variable.name="method")
  }else if(trend == "day"){
    mdf <- melt(dat, id.vars="day", value.name="MSE", variable.name="method")
  }else{
    mdf <- melt(dat, id.vars="Time", value.name="MSE", variable.name="method")
  }
  data_targets[[target]] = mdf
}

# plotting ----------------------------------------------------------------

gg_one_target <-function(dat, threshold, x_axis = T, y_axis = T){
  n_est = length(levels(unique(data_targets[[l]][,2])))
  if (trend == "sd_R") {
    g = ggplot(data=dat, aes(x=sd_R, y=MSE, group = method, colour = method))
  }else if(trend == "day"){
    g = ggplot(data=dat, aes(x=day, y=MSE, group = method, colour = method))
  }else{
    g = ggplot(data=dat, aes(x=Time * 48, y=MSE, group = method, colour = method))
  }

  point_size = 1
  g = g + geom_line() +
    geom_point( size = point_size, aes(shape = method), fill="white")+ 
    scale_shape_manual(values=c(1:n_est)) +
    ggtitle(paste(sep = "", "c = ", threshold)) + # paste(title_prefix, name[l] ,sep = " ") # Threshold
    theme(plot.title = element_text(hjust = 0.5))
 
  g = g + ylim(0, NA)
   
  # + 
    # scale_x_continuous(breaks = seq(7, 11, 1)) + 
    # scale_y_continuous(breaks = seq(0, 12, 1))
  if (x_axis) {
   
    if (trend == "sd_R") {
      g = g + xlab( TeX('$\\sigma_R$'))
    }else if(trend == "day"){
      g = g + xlab("Number of Days")
    }else{
      g = g + xlab("T")
    }
  }
  if (y_axis) {
    g = g + ylab("MSE")
  }else{
    g = g + ylab(NULL) # ,axis.ticks.x=element_blank()
  }
  
  
  return(g)
}

ggs = list()
for (l in c(1:target_num)) {
  ggs[[l]] = gg_one_target(data_targets[[l]], threshold_levels[l], y_axis = F) + 
    theme(plot.margin = unit(c(0.1, 0.1, 0.1, -.1), "cm"))
}

g = ggarrange(ggs[[1]], ggs[[2]], ggs[[3]], ggs[[4]], #ggs[[5]], ggs[[6]], # ggs[[7]], ggs[[8]]
              ncol = 4, nrow = 1, 
              # legend = "none",
              common.legend = TRUE, legend="bottom",
              align = "v")# , ggs[[3]], ggs[[4]]

g1 = annotate_figure(g,
                left = text_grob(y_var, rot = 90, hjust = 0))
# save
print(g1)
ggsave( filename = paste(sep = "", "/Users/mac/Desktop/", y_var, ".png"),  plot = g1,
        width = 6, height = 2.5,  units = c("in"), dpi = 1000 )
