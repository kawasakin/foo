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
