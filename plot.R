library("ggplot2")

# caption; legend; comparison; x and y axis
one_plot <-function(){
  gg = ggplot(data, aes(x_date, count))+
    geom_line(data=obs, col = colours[1], na.rm = T)
  
  gg = gg + geom_line(data = est,
                      aes(x = x_date, y = lower),
                      col = colours[3], linetype = 'dotted') +
    geom_line(data = est,
              aes(x = x_date, y = upper),
              col = colours[3], linetype = 'dotted') +
    geom_line(data = est,
              aes(x = x_date, y = mean),
              col = colours[4], linetype = 'solid') +
    ggtitle(paste(title_prefix, name[l] ,sep = " ")) +
    xlab("Date") + ylab("count")+
    theme(plot.title = element_text(hjust = 0.5))
  
  return()
    
}



if (plot) {
  grid.arrange(gg_I, gg_R,  gg_new, nrow = 3, top = textGrob(main_title, gp=gpar(fontsize=15,font=1)))
}