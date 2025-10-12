library(plyr)
library(ggplot2)
library(reshape2)

# load data
data = read.csv("./results/simulation.csv")
data = data[, -c(1, 3)]

##################### MSE Comparison ##################### 
MSE_data = data[, c("N", "CD_mse", "DS_mse", "GPA_CD_mse", "GPA_DS_mse")]
MSE_data$N = as.factor(MSE_data$N)
MSE_data = melt(MSE_data, id=c("N"), 
                variable.name = "Estimator", 
                value.name = "MSE")
MSE_data$Estimator = factor(MSE_data$Estimator, 
                            levels = c('CD_mse', 'DS_mse', 'GPA_CD_mse', 'GPA_DS_mse'),
                            labels = c("CD", "DS", "GPA-CD", "GPA-DS"))
str(MSE_data)


par(mfrow = c(1, 1), oma=c(0.2,0.2,0.2,0.2))
plot_box = function(thres, times){
  boxplot(log(MSE)~Estimator, MSE_data[MSE_data$N == thres, ], 
          xlab="",
          ylab="",
          ylim = c(-5.5, -1.5),
          main=paste0('N=', thres),
          cex.lab = times,
          cex.axis = times,
          cex.main=times
  )
}

pdf("./results/logMSE_N=100.pdf") # create painting environment
plot_box(100, 1.5) # boxplot
dev.off() # close the environment

pdf("./results/logMSE_N=500.pdf") # create painting environment
plot_box(500, 1.5) # boxplot
dev.off() # close the environment

pdf("./results/logMSE_N=1000.pdf") # create painting environment
plot_box(1000, 1.5) # boxplot
dev.off() # close the environment

##################### Time Comparison ##################### 
time_data = data[, c("N", "CD_time", "DS_time", "GPA_CD_time", "GPA_DS_time")]
time_data$N = as.factor(time_data$N)
time_data = melt(time_data, id=c("N"), 
                variable.name = "Estimator", 
                value.name = "Time")
time_data$Estimator = factor(time_data$Estimator, 
                            levels = c('CD_time', 'DS_time', 'GPA_CD_time', 'GPA_DS_time'),
                            labels = c("CD", "DS", "GPA-CD", "GPA-DS"))
str(time_data)


par(mfrow = c(1, 1), oma=c(0.2,0.2,0.2,0.2))
plot_box = function(thres, times){
  boxplot(Time~Estimator, time_data[time_data$N == thres, ], 
          ylim = c(0, 2),
          xlab="",  ylab="",
          # ylab="Avg Time Cost",
          cex.lab = times,
          cex.axis = times,
          main=paste0('N=', thres),
          cex.main=times)
}
pdf("./results/time_N=100.pdf") # create painting environment
plot_box(100, 1.5) # boxplot
dev.off() # close the environment

pdf("./results/time_N=500.pdf") # create painting environment
plot_box(500, 1.5) # boxplot
dev.off() # close the environment

pdf("./results/time_N=1000.pdf") # create painting environment
plot_box(1000, 1.5) # boxplot
dev.off() # close the environment
