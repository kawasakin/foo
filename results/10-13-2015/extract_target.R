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

pairs = read.table("loops.300K.3E.rep2.txt")
genes = read.table("genes.txt")
enhancers = read.table("enhancers.2K.bed")
res = data.frame(genes[pairs[,1],], enhancers[pairs[,2],], pairs)
res = res[order(res[,6]),]
write.table(res, file = "loops.300K.3E.rep2.txt", append = FALSE, quote = FALSE, sep = "\t",
            eol = "\n", na = "NA", dec = ".", row.names = FALSE,
            col.names = FALSE, qmethod = c("escape", "double"),
            fileEncoding = "")


