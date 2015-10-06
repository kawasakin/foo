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


# loops
library(GenomicRanges)
num_genes = 11041
data = read.table("../results/09-25-2015/loops_pred.txt")
num_rep = nrow(data)/num_genes
data$id = rep(1:num_genes, num_rep)

a1 = data.frame(data$id, data[,1]+1)
a2 = data.frame(data$id, data[,2]+1)
a3 = data.frame(data$id, data[,3]+1)
a4 = data.frame(data$id, data[,4]+1)

a1 = a1[which(a1[,2]>0),]
a2 = a2[which(a2[,2]>0),]
a3 = a3[which(a3[,2]>0),]
a4 = a4[which(a4[,2]>0),]

colnames(a1) = colnames(a2) = colnames(a3) = colnames(a4) = c("id", "e")
a = rbind(a1, a2, a3, a4)
colnames(a) = c("p", "e")
a = a[order(a$p),]
freq = table(paste(a[,1], a[,2]))
loops = names(freq[which(freq>=0)])
a = data.frame(do.call(rbind, strsplit(loops, split=" ")))
a$freq = freq/num_rep
write.table(a, file = "a", append = FALSE, quote = FALSE, sep = "\t",
            eol = "\n", na = "NA", dec = ".", row.names = FALSE,
            col.names = FALSE, qmethod = c("escape", "double"),
            fileEncoding = "")
a = read.table("a") 
colnames(a) = c("p", "e", "prob")
a = a[order(a$p),]

b = data.frame(a = 1:34729, prob = c(rep(0, 2133), a$prob))
m <- ggplot(b, aes(x = prob))
m + geom_density(colour="darkgreen", size=1, fill="green", adjust=1/3.5)


b = read.table("../results/09-25-2015/matches_loops.txt")
b = b + 1
colnames(b) = c("p", "e")

# chose those that overlap with b
a = a[which(a$p %in% b$p),]
a$be = b$e[match(a$p, b$p)]

promoters = read.table("../results/09-25-2015/gene_3k_promoter.bed")
enhancers = read.table("../results/09-25-2015/enhancers.2K.bed")


nrow(a[which(a[,1]<4431),])/4431/17
nrow(a[which(a[,1]>=4431),])/6610/17

####################
data = read.table("../results/09-25-2015/res_enh.txt")
res = data.frame()
for(i in 0:49*15+5){
	res = rbind(res, data[i:(i+10),])
}
boxplot(V2~V1, res, outline=FALSE, ylab="Cost")
boxplot(V3~V1, res, outline=FALSE, ylab="Accr")





