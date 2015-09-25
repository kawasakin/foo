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
data = t(read.table("tmp.txt"))
data = data.frame(data[order(data[,2]),])
data$label = 2
data$label[1:367] = 1
data$X2 = 1:nrow(data)
color_transparent <- adjustcolor(data$label, alpha.f = 0.3) 
plot(data$X2, data$X1, col=color_transparent, pch=19, xlab="", xaxt='n', ann=FALSE)

