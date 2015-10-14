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
data = read.table("../results/09-25-2015/loops_pred.300k.raw.rep1.txt")
num_rep = nrow(data)/num_genes
data$id = rep(1:num_genes, num_rep)
id = c()
for(i in 1:num_rep){
	id = c(id, rep(i, num_genes))
}
data$id = id
data.list <- split(data, data$id)
res = data.frame()
for(i in 1:length(data.list)){
	a = data.list[[i]]
	a1 = data.frame(1:num_genes, a[,1]+1)
	a2 = data.frame(1:num_genes, a[,2]+1)
#	a3 = data.frame(1:num_genes, a[,3]+1)
#	a4 = data.frame(1:num_genes, a[,4]+1)

	a1 = a1[which(a1[,2]>0),]
	a2 = a2[which(a2[,2]>0),]
#	a3 = a3[which(a3[,2]>0),]
#	a4 = a4[which(a4[,2]>0),]

#	colnames(a1) = colnames(a2) = colnames(a3) = colnames(a4) = c("id", "e")
	colnames(a1) = colnames(a2) = c("id", "e")

#	a = rbind(a1, a2, a3, a4)
	a = rbind(a1, a2)

	colnames(a) = c("p", "e")
	a = a[order(a$p),]
	res = rbind(res, c(nrow(a[which(a$p<=4431),])/4431, nrow(a[which(a$p>4431),])/6610))
}
colnames(res) = c("silent", "active")
boxplot(res)

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
write.table(a, file = "loops.rep2.txt", append = FALSE, quote = FALSE, sep = "\t",
            eol = "\n", na = "NA", dec = ".", row.names = FALSE,
            col.names = FALSE, qmethod = c("escape", "double"),
            fileEncoding = "")

rep2 = read.table("loops.rep2.txt") 
colnames(rep2) = c("p", "e", "prob")
rep2 = rep2[order(rep2$p),]

rep1 = read.table("loops.rep1.txt") 
colnames(rep1) = c("p", "e", "prob")
rep1 = rep1[order(rep1$p),]

rep1 = rep1[which(paste(rep1[,1], rep1[,2]) %in% paste(rep2[,1], rep2[,2])),]
rep2 = rep2[which(paste(rep2[,1], rep2[,2]) %in% paste(rep1[,1], rep1[,2])),]

plot(rep1$prob, rep2$prob, pch=19, cex=0.4, xlab="rep1", ylab="rep2", col=rgb(0,0,0,alpha=0.3))

rep1 = rep1[rep1$prob>0.8,]
rep2 = rep2[rep2$prob>0.8,]
rep1 = rep1[which(paste(rep1[,1], rep1[,2]) %in% paste(rep2[,1], rep2[,2])),]
rep2 = rep2[which(paste(rep2[,1], rep2[,2]) %in% paste(rep1[,1], rep1[,2])),]

num_overlap = length(which(paste(rep1[,1], rep1[,2]) %in% paste(rep2[,1], rep2[,2])))
num_overlap/nrow(rep1)
num_overlap/nrow(rep2)


library(ggplot2)
b = data.frame(a = 1:34729, prob = c(rep(0, 34729-nrow(a)), a$prob))
m <- ggplot(b, aes(x = prob))
m + geom_density(colour="darkgreen", size=1, fill="green")


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
data = read.table("../results/09-25-2015/res_enh.rep1.txt")
data_random = read.table("../results/09-25-2015/res_enh.random.txt")

res = data.frame()
res_random = data.frame()
for(i in 0:49*15+5){
	res = rbind(res, data[i:(i+10),])
	res_random = rbind(res_random, data_random[i:(i+10),]) 
}

res_random = do.call(rbind, lapply(split(res_random, res_random$V1), colMeans))
res_random[,1] = res_random[,1] + 1

boxplot(V2~V1, res, outline=FALSE, ylab="Cost", ylim=c(0.15, 0.35))
lines(x = res_random[,1], y = res_random[,2],  col="red")
points(x = res_random[,1], y = res_random[,2], pch=19, cex=0.3,  col="red")

boxplot(V3~V1, res, outline=FALSE, ylab="Accr", ylim=c(0.86, 0.96))
lines(x = res_random[,1], y = res_random[,3], col="red")
points(x = res_random[,1], y = res_random[,3], pch=19, cex=0.3, col="red")

data = read.table("../results/09-27-2015/res.txt")
par(mfrow = c(1, 2))
plot(1:nrow(data), data[,3], pch=19, cex=0.4, ylab="", xlab="")
points(1:nrow(data), data[,5], pch=18, col="red",cex=0.4)

plot(1:nrow(data), data[,2], pch=19, cex=0.4, ylab="", xlab="")
points(1:nrow(data), data[,4], pch=18, col="red",cex=0.4)

plot(1:10, data[1:10,2], pch=19)
points(1:10, data[1:10,4], pch=18)

data = read.table("../results/09-27-2015/res.txt")


num_genes = 11041
data = read.table("../results/09-25-2015/matches_distance.txt")
freq = data.frame(table(data[,1]))
freq2 = data.frame(Var1=0:(num_genes-1), Freq=0)
freq2$Freq[match(freq$Var1, freq2$Var1)] = freq$Freq
freq2$group = c(rep("silence", 4431), rep("active",6610))
boxplot(Freq~group, freq2, outline=FALSE)


