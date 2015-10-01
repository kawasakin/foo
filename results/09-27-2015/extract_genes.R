library(mixtools)
 
options(echo=TRUE) # if you want see commands in output file
args <-commandArgs(TRUE)
print(options)

flanking_up_len = as.integer(args[1])
flanking_down_len = as.integer(args[2])
out_file = args[3]

file_gene = "/oasis/tscc/scratch/r3fang/github/foo/results/09-12-2015/mESC-zy27.gene.expr.sel"
flanking_up_len = 3000
flanking_down_len = 0
out_file="promoters.bed"

get_promoter <- function(genes, upstream_region=3000, downstream_region=2000){
	# split data by the strand
	genes.list = split(genes, genes$strand)
	res.list <- lapply(genes.list, function(x){
		if(as.character(x$strand[1])=="+"){
			x$end = x$start + downstream_region
			x$start = x$start - upstream_region
		}else{
			x$start = x$end - downstream_region
			x$end = x$end + upstream_region
		}
		return(x)
	})
	res = data.frame(do.call(rbind, res.list))
	res = res[order(res$index),]
	return(res)
}

genes = read.table(file_gene, head=T)
genes$FPKM = log(genes$FPKM)
genes = genes[order(genes$FPKM),]
genes = rbind(genes[1:4431,], genes[(nrow(genes)-6609):nrow(genes),])
a = data.frame(cut(genes$FPKM[1:4331], 4, labels=c(0, 0.1, 0.2, 0.3)))
b = data.frame(cut(genes$FPKM[4332:nrow(genes)], 5, labels=c(0.6, 0.7, 0.8, 0.9, 1)))
colnames(a) = colnames(b) = "label"

write.table(data.frame(rbind(a,b)), file ="datY.dat", append = FALSE, quote = FALSE, sep = "\t",
eol = "\n", na = "NA", dec = ".", row.names = FALSE,
col.names = FALSE, qmethod = c("escape", "double"),
fileEncoding = "")

genes = genes[,c(3,4,5,6,9)]
colnames(genes) = c("chr", "start", "end", "FPKM", "strand")

genes = genes[order(genes$FPKM),]
genes$index = 1:nrow(genes)
promoters <- get_promoter(genes, flanking_up_len, flanking_down_len)
promoters$start[which(promoters$start<0)] = 1

write.table(promoters, file ="promoters.bed", append = FALSE, quote = FALSE, sep = "\t",
eol = "\n", na = "NA", dec = ".", row.names = FALSE,
col.names = FALSE, qmethod = c("escape", "double"),
fileEncoding = "")



