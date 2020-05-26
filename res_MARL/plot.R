library("ggplot2")
library("reshape2")
library("ggpubr")
library("latex2exp")
library("readr")

# Data --------------------------------------------------------------------




# Data --------------------------------------------------------------------
res = list()
path_plot = "/Users/mac/Google Drive/CausalMARL/res_MARL/temp_py/"

## variable to plot 
y_var = "MSE"
# y_var = "bias"
# y_var = "std"
# y_var = "sd_MSE"

## the trend variable for x-axis
# x.trend = "sd_R"
x.trend = "Time"
# x.trend = "day"

trend = x.trend
if (trend == "sd_R") {
  setting_range = seq(0, 30, 5)
  for (i in 1:length(setting_range)) {# x-axis
    path = paste(sep = "", path_plot, "res_sd_", y_var, i, ".txt")
    res_1 <- read_csv(path,
                      col_names = FALSE)
    # res_1 = res_1[c(3,4,6,7),]
    res[[i]] = abs(res_1)
  }
}else{
  setting_range = c(2, 3, 4, 5, 6, 7, 8)
  for (i in 1:length(setting_range)) {# x-axis
    path = paste(sep = "", path_plot, "res_T_", y_var, i, ".txt")
    res_1 <- read_csv(path,
                      col_names = FALSE)
    # res_1 = res_1[c(3,4,6,7),]
    res[[i]] = abs(res_1)
  }
  
}



# if (trend == "sd_R") {
#   res = readRDS("/Users/mac/Google Drive/CausalMARL/res_MARL/final/final_sd/final_sd.RDS")
# }else{
#   res = readRDS("/Users/mac/Google Drive/CausalMARL/res_MARL/final_0421/final_T/final_T.RDS")
# }



# saveRDS(res, "/Users/mac/Google Drive/CausalMARL/res_MARL/final_0421/final_T/final_T.RDS")

## number of sub-plots for different target policies
target_num = 4
threshold_levels= c(9, 8, 7, 6)


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


# Transform tables to NIPS Readme -----------------------------------------

transRawTablesToFinalReadme <- function(data_targets){
  table.readme = list()
  for (target in 1:target_num){
    table.readme[[target]] = data_targets[[target]]
    table.readme[[target]][["target"]] = threshold_levels[target]
  }
  return(do.call(rbind, table.readme))
}

table.readme = transRawTablesToFinalReadme(data_targets)

saveRDS(table.readme, paste(sep = "", "readme_table_", x.trend, ".rds"))


# plotting ----------------------------------------------------------------

gg_one_target <-function(dat, threshold,  y_axis = T){
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
    scale_shape_manual(values=c(1:n_est))  + # paste(title_prefix, name[l] ,sep = " ") # Threshold
    theme(plot.title = element_text(hjust = 0.5))
  
  if(trend =="sd_R")g = g + ggtitle(paste(sep = "", "K = ", threshold)) + theme(plot.title = element_text(size = 10)) # , face = "bold"
  
  g = g + ylim(0, NA)
  
  if (trend == "sd_R") {
      g = g + xlab( TeX('$\\sigma_R$'))
    }else if(trend == "day"){
      g = g + xlab("Number of Days")
    }else{
      g = g + xlab("T")
      g = g + xlim(NA, 400)
      # g = g + scale_x_continuous(breaks = seq(100, 400, 100))
  }
  g = g + ylab(NULL) 
  # if (y_axis) {
  #   g = g + ylab("MSE", element_text(size = 1222)) + theme(axis_title_y = element_text(size=1222, text='beef'))
  # }else{g = g + ylab(NULL) # ,axis.ticks.x=element_blank()
  # }
  # g = g + font("ylab", size = 120)
  return(g)
}


### Combine plots --------

ggs = list()
for (l in c(1:target_num)) {
  ggs[[l]] = gg_one_target(data_targets[[l]], threshold_levels[l], y_axis = F) + 
    theme(plot.margin = unit(c(0.1, 0.1, 0.1, -0.1), "cm"))
}




if(trend == "Time"){
  g = ggarrange(ggs[[1]], ggs[[2]], ggs[[3]], ggs[[4]], # ggs[[5]], ggs[[6]], # ggs[[7]], ggs[[8]]
                ncol = 4, nrow = 1, 
                legend = "bottom",
                common.legend = T, 
                align = "v")# , ggs[[3]], ggs[[4]]
  g1 = annotate_figure(g, left = text_grob(y_var, just ="centre",  
                                           size = 10, rot = 90, hjust = -0.85, vjust = 0))# 
  
}else{
  g = ggarrange(ggs[[1]], ggs[[2]], ggs[[3]], ggs[[4]], # ggs[[5]], ggs[[6]], # ggs[[7]], ggs[[8]]
                ncol = 4, nrow = 1, 
                legend = "none",
                align = "v")# , ggs[[3]], ggs[[4]]
  g1 = annotate_figure(g, left = text_grob(y_var, just ="centre",  
                                           size = 10, rot = 90, hjust = 0, vjust = 0))# 
}
# save
print(g1)
# _NoNaive
if (trend == "sd_R") {
  ggsave( filename = paste(sep = "", "/Users/mac/Desktop/", y_var, "_sd_NoNaive.png"),  plot = g1,
          width = 6.5, 
          height = 2.1,  units = c("in"), dpi = 1000 )
}else{
  ggsave( filename = paste(sep = "", "/Users/mac/Desktop/", y_var, "_T_NoNaive.png"),  plot = g1,
          width = 6.5, 
          height = 2.2,  units = c("in"), dpi = 1000 )
}
