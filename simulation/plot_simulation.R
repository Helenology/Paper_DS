# ============================================================
# Load required libraries
# ============================================================
library(plyr)       # Provides advanced data manipulation tools
library(reshape2)   # Used to reshape data frames (wide ↔ long format)

# ============================================================
# Load simulation data
# ============================================================
data = read.csv("./results/simulation.csv")


# ============================================================
# Part 1: MSE (Mean Squared Error) Comparison
# ============================================================

# Select relevant columns for MSE comparison
MSE_data = data[, c("N", "CD_mse", "DS_mse", "GPA_CD_mse", "GPA_DS_mse")]

# Convert N to a factor (categorical variable)
MSE_data$N = as.factor(MSE_data$N)

# Reshape the data from wide format to long format
MSE_data = melt(MSE_data, id=c("N"), 
                variable.name = "Estimator", 
                value.name = "MSE")

# Rename estimator labels for better readability in plots
MSE_data$Estimator = factor(MSE_data$Estimator, 
                            levels = c('CD_mse', 'DS_mse', 'GPA_CD_mse', 'GPA_DS_mse'),
                            labels = c("CD", "DS", "GPA-CD", "GPA-DS"))

# Define a reusable boxplot function for MSE comparison
par(mfrow = c(1, 1), oma=c(0.2,0.2,0.2,0.2))
plot_box = function(thres, times){
  # Plot a boxplot of log(MSE) for a given sample size (N = thres)
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

# Generate MSE boxplots for different sample sizes
pdf("./results/logMSE_N=100.pdf") # Open PDF device
plot_box(100, 1.5)                # Plot for N = 100
dev.off()                         # Close device

pdf("./results/logMSE_N=500.pdf") # Open PDF device
plot_box(500, 1.5)                # Plot for N = 500
dev.off()                         # Close device

pdf("./results/logMSE_N=1000.pdf") # Open PDF device
plot_box(1000, 1.5)                # Plot for N = 1000
dev.off()                          # Close device


# ============================================================
# Part 2: Computation Time Comparison
# ============================================================

# Select relevant columns for time comparison
time_data = data[, c("N", "CD_time", "DS_time", "GPA_CD_time", "GPA_DS_time")]

# Convert N to factor
time_data$N = as.factor(time_data$N)

# Reshape from wide to long format
time_data = melt(time_data, id=c("N"), 
                variable.name = "Estimator", 
                value.name = "Time")

# Rename estimators
time_data$Estimator = factor(time_data$Estimator, 
                            levels = c('CD_time', 'DS_time', 'GPA_CD_time', 'GPA_DS_time'),
                            labels = c("CD", "DS", "GPA-CD", "GPA-DS"))

# Define boxplot function for time cost comparison
par(mfrow = c(1, 1), oma=c(0.2,0.2,0.2,0.2))
plot_box = function(thres){
  # Plot a boxplot of log(Time) for a given sample size (N = thres)
  boxplot(log(Time)~Estimator, time_data[time_data$N == thres, ], 
          ylim = c(-8, 1),
          xlab="",  ylab="",
          cex.lab = 1.5,
          cex.axis = 1.5,
          main=paste0('N=', thres),
          cex.main=1.5)
}

# Generate time comparison boxplots for different sample sizes
pdf("./results/logtime_N=100.pdf")  # Open PDF device
plot_box(100)                       # Plot for N = 100
dev.off()                           # Close device

pdf("./results/logtime_N=500.pdf")  # create painting environment
plot_box(500)                       # Plot for N = 500
dev.off()                           # Close device

pdf("./results/logtime_N=1000.pdf") # Open PDF device
plot_box(1000)                      # Plot for N = 1000
dev.off()                           # Close device
