library(GenomicRanges)
num_genes = 11041
data = read.table("loops.300K.3E.raw.rep1.txt")
num_rep = nrow(data)/num_genes
data$id = rep(1:num_genes, num_rep)
a1 = data.frame(data$id, data[,1]+1)
a2 = data.frame(data$id, data[,2]+1)
a3 = data.frame(data$id, data[,3]+1)

a1 = a1[which(a1[,2]>0),]
a2 = a2[which(a2[,2]>0),]
a3 = a3[which(a3[,2]>0),]

colnames(a1) = colnames(a2) = colnames(a3) = c("id", "e")
a = rbind(a1, a2, a3)
colnames(a) = c("p", "e")
a = a[order(a$p),]
freq = data.frame(table(paste(a[,1], a[,2])))
freq$Freq = freq$Freq/num_rep

write.table(freq, file = "loops.300K.3E.rep1.txt", append = FALSE, quote = FALSE, sep = "\t",
            eol = "\n", na = "NA", dec = ".", row.names = FALSE,
            col.names = FALSE, qmethod = c("escape", "double"),
            fileEncoding = "")

loops = read.table("loops.300K.3E.rep1.txt")
loops = loops[order(loops[,1]),]
genes = read.table("genes.txt")
genes = genes[,c(1,3,4,5,6,9,10)]
enhancers = read.table("enhancers.2K.bed")
targets = cbind(genes[loops[,1],], enhancers[loops[,2],], loops)

colnames(targets) = c("gname", "gchr", "gstart", "gend", "FPRM", "strand", "glabel", "echr", "estart", "eend", "gid", "eid", "prob")
write.table(targets, file = "loops.300K.3E.rep1.txt", append = FALSE, quote = FALSE, sep = "\t",
            eol = "\n", na = "NA", dec = ".", row.names = FALSE,
            col.names = FALSE, qmethod = c("escape", "double"),
            fileEncoding = "")

