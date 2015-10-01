options(echo=TRUE) # if you want see commands in output file
args <-commandArgs(TRUE)
print(options)

flanking_up_len = as.integer(args[1])
flanking_down_len = as.integer(args[2])
out_file = args[3]

file_gene = "/oasis/tscc/scratch/r3fang/github/foo/results/09-12-2015/mESC-zy27.gene.expr.sel"
flanking_up_len   = 50000
flanking_down_len = 50000
out_file="gene_100k_flanking.bed"

get_promoter <- function(genes, upstream_region=3000, downstream_region=2000){
	# split data by the strand
	genes.list = split(genes, genes$strand)
	res.list <- lapply(genes.list, function(x){
		if(as.character(x$strand[1])=="+"){
			x$end   =   x$start + downstream_region
			x$start = x$start - upstream_region
		}else{
			x$start = x$end - downstream_region
			x$end   = x$end + upstream_region
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
genes$label = c(rep(0, 4431), rep(1, 6610))
genes = genes[,c(3, 4, 5, 6, 9, 10)]
colnames(genes) = c("chr", "start", "end", "FPKM", "strand", "label")
genes = genes[order(genes$FPKM),]
genes$index = 1:nrow(genes)
promoters <- get_promoter(genes, flanking_up_len, flanking_down_len)
promoters$start[which(promoters$start<0)] = 1

write.table(promoters[,1:6], file =out_file, append = FALSE, quote = FALSE, sep = "\t",
eol = "\n", na = "NA", dec = ".", row.names = FALSE,
col.names = FALSE, qmethod = c("escape", "double"),
fileEncoding = "")


