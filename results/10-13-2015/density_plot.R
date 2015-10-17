# density plot
library(ggplot2)
data = read.table("loops.300K.3E.rep1.txt")
data = data[which(data$V16>0.1),]
m <- ggplot(data, aes(x = V16))
m + geom_density(colour="black", size=1, adjust=0.3) +
theme_bw()+theme(legend.position="none", axis.title.x=element_blank(),axis.title.y=element_blank())
------------------------------------------------------------------------------------
# points plot for replicates
library(ggplot2)
n = read.table("loops.rep1.sel.txt", head=TRUE)
rep2 = read.table("loops.rep2.sel.txt", head=TRUE)

rep1 = rep1[which(paste(rep1$gid, rep1$eid) %in% paste(rep2$gid, rep2$eid)),]
rep2 = rep2[which(paste(rep2$gid, rep2$eid) %in% paste(rep1$gid, rep1$eid)),]
data = data.frame(cbind(rep1=rep1$log.p.value.correct, rep2=rep2$log.p.value.correct))
m <- ggplot(data[sample(1:nrow(data), 5000),], aes(x = rep1, y= rep2))
m + geom_point(aes(alpha=0.01, colour="red")) + theme_bw()+
theme(legend.position="none", axis.title.x=element_blank(),
axis.title.y=element_blank())

rep1 = read.table("loops.300K.3E.rep1.txt")
rep2 = read.table("loops.300K.3E.rep2.txt")
rep1 = rep1[which(rep1[,16]>0.85),]
rep2 = rep2[which(rep2[,16]>0.85),]
length(which(paste(rep1[,14], rep1[,15]) %in% paste(rep2[,14], rep2[,15])))
length(which(paste(rep2[,14], rep2[,15]) %in% paste(rep1[,14], rep1[,15])))

------------------------------------------------------------------------------------
# distance distribution (d)
library(ggplot2)
rep1 = read.table("loops.rep1.sel.txt", head=TRUE)
rep1$dist = apply(data.frame(abs(rep1$gstart - rep1$estart), abs(rep1$gstart - rep1$end), abs(rep1$gend - rep1$end), abs(rep1$gend - rep1$estart)), 1, min)

m <- ggplot(data=rep1, aes(dist))
m +  geom_histogram(col="black", alpha = .2, binwidth = 5000)+
theme_bw()+theme(legend.position="none", axis.title.x=element_blank(),
				 axis.title.y=element_blank(), axis.text.y = element_text(angle = 45, hjust=0.5))+xlim(0, 150000)
matches = read.table("matches_distance.txt")
matches = matches + 1
enhancers = read.table("enhancers.2K.bed")
promoters = read.table("genes.txt")
data = cbind(promoters[matches[,1],], enhancers[matches[,2],])
data$dist = apply(data.frame(abs(data[,4] - data[,12]), abs(data[,4] - data[,13]), abs(data[,5] - data[,12]), abs(data[,5] - data[,13])), 1, min)
