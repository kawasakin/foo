# fig2
library(mixtools)
data = read.table("../results/09-12-2015/dat_Y.txt")
mixmdl = normalmixEM(data[,1])
plot(mixmdl,which=2, breaks=100, xlim=c(-8, 8))
lines(density(data[,1]),lty=2, lwd=2)

# group 0
length(which(data[,1] <= (mixmdl$mu - 2*mixmdl$sigma)[1]))
# group 1 
length(which(data[,1] >= (mixmdl$mu + 2*mixmdl$sigma)[2]))

# threshold is -0.7664458 and 1.575

#fit3 leanring rate

#fig3
data = t(read.table("../results/09-23-2015/teY_prob.txt"))
data = data.frame(t(data))
#data = data.frame(data[order(data[,2]),])
data$V1[which(data$V1==1)] = 2
data$V1[which(data$V1==0)] = 1
data$X2 = 1:nrow(data)
color_transparent <- adjustcolor(data$V1, alpha.f = 0.3) 
plot(data$X2, data$V2, col=color_transparent, pch=19, xlab="", xaxt='n', ann=FALSE)


n = 100
a = rnorm(n, mean = 0.35, sd = 0.04)
b = rnorm(n, mean = 0.27, sd = 0.04)
c = rnorm(n, mean = 0.25, sd = 0.02)
d = rnorm(n, mean = 0.23, sd = 0.01)
boxplot(a, b, c, d)