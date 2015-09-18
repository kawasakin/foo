#PBS -q hotel 
#PBS -N HTseq_count
#PBS -l nodes=5:ppn=4
#PBS -l walltime=70:00:00
#PBS -o HTseq_count.out
#PBS -e HTseq_count.err
#PBS -V
#PBS -M r3fang@ucsd.edu
#PBS -m ae
#PBS -A ren-group
###### qsub -I -q hotel -N GeneExp -l nodes=1:ppn=1 -l walltime=100:00:00 -A ren-group 

# constant
RNA_SEQ_DIR="/oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/ENCODE/RNA-seq/"
GENOME="/oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/Sequence/WholeGenomeFasta/genome.fa"
GTF="/oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/Annotation/Genes/genes.gtf"

cd /oasis/tscc/scratch/r3fang/github/foo/results/09-12-2015/

# 0. download gene expression data
# wget http://chromosome.sdsc.edu/mouse/download/19-tissues-expr.zip 
# unzip 19-tissues-expr.zip 
# cp 19-tissues-expr/mESC-* ./
# rm -r 19-tissues-expr
# rm 19-tissues-expr.zip

# 1. process ,gtf file
cat $GTF | awk '{print $1, $7, $16}' - | sort - | uniq - > genes.gtf 
# replace " and ;  with nothing 

# -------------------------------------------------------------------------------------------
# 2. get strand infor for expression data
# R: 
data = read.table("mESC-zy28.gene.expr", head=T)
genes = read.table("genes.gtf")
colnames(genes) = c("chr", "strand", "gene_id")
names <- unique(intersect(data$gene_id, genes$gene_id))
data.sel <- data[which(data$gene_id %in% names),]
genes.sel <- genes[which(genes$gene_id %in% names),]
data.sel$strand = genes.sel$strand[match(data.sel$gene_id, genes.sel$gene_id)]
write.table(data.sel, file = "mESC-zy28.gene.expr.sel", append = FALSE, quote = FALSE, sep = "\t",
                 eol = "\n", na = "NA", dec = ".", row.names = FALSE,
                 col.names = TRUE, qmethod = c("escape", "double"),
                 fileEncoding = "")
# -------------------------------------------------------------------------------------------

#3. get features
library(GenomicRanges)
library(parallel)

data = read.table("mESC-zy27.gene.expr.sel", head=T)
data$FPKM = log(data$FPKM)
upstream_region = 3000
downstream_region = 2000
bin_size = 50
bin_num = (upstream_region+downstream_region)/bin_size
#data.sel$label = cut(data.sel$FPKM, include.lowest = TRUE, quantile(data.sel$FPKM, probs = seq(0, 1, 1/num_of_exp)), labels=1:num_of_exp)

# split data by the strand
data.list = split(data, data$strand)
res.list <- lapply(data.list, function(x){
	if(as.character(x$strand[1])=="+"){
		x$right = x$left + downstream_region
		x$left = x$left - upstream_region
	}else{
		x$left = x$right - downstream_region
		x$right = x$right + upstream_region
	}
	return(x)
})

promoters = data.frame(do.call(rbind, res.list))
promoters = promoters[order(promoters$FPKM),]
promoters$index = 1:nrow(promoters)
promoters.list = split(promoters, promoters$index)

res.list <- mclapply(promoters.list, function(x){
	if(x$strand=="+"){
		y = data.frame(chr=x$chr, start=seq(as.numeric(x$left), as.numeric(x$right), by=bin_size)[1:bin_num-1], end= seq(as.numeric(x$left), as.numeric(x$right), by=bin_size)[2:bin_num], gene_id=x$index, index=1:(bin_num-1), strand=x$strand)
	}else{
		y = data.frame(chr=x$chr, start = seq(as.numeric(x$right), as.numeric(x$left), by=-bin_size)[2:bin_num], end =seq(as.numeric(x$right), as.numeric(x$left), by= -bin_size)[1:bin_num-1], gene_id=x$index, index=1:(bin_num-1), strand=x$strand)		
	}
	return(y)
}, mc.cores=10)

bins = data.frame(do.call(rbind, res.list))

colnames(bins) = c("chr", "start", "end", "gene_id", "bin_id", "strand")
bins.gr <- with(bins, GRanges(chr, IRanges(start+1, end), strand="*", gene_id, bin_id))

file.names = c(
"/oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/ENCODE/ChIP-seq/E14_CHD2.bed",
"/oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/ENCODE/ChIP-seq/E14_HCFC1.bed",
"/oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/ENCODE/ChIP-seq/E14_MAFK.bed",
"/oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/ENCODE/ChIP-seq/E14_NANOG.bed",
"/oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/ENCODE/ChIP-seq/E14_POU5F1.bed",
"/oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/ENCODE/ChIP-seq/E14_ZC3H11A.bed",
"/oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/ENCODE/ChIP-seq/E14_ZNF384.bed",
"/oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/ENCODE/ChIP-seq/wgEncodeLicrHistoneEsb4H3k09acME0C57bl6StdPk.broadPeak",
"/oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/ENCODE/ChIP-seq/wgEncodeLicrHistoneEsb4H3k09me3ME0C57bl6StdPk.broadPeak",
"/oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/ENCODE/ChIP-seq/wgEncodeLicrHistoneEsb4H3k27acME0C57bl6StdPk.broadPeak",
"/oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/ENCODE/ChIP-seq/wgEncodeLicrHistoneEsb4H3k27me3ME0C57bl6StdPk.broadPeak",
"/oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/ENCODE/ChIP-seq/wgEncodeLicrHistoneEsb4H3k36me3ME0C57bl6StdPk.broadPeak",
"/oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/ENCODE/ChIP-seq/wgEncodeLicrHistoneEsb4H3k4me1ME0C57bl6StdPk.broadPeak",
"/oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/ENCODE/ChIP-seq/wgEncodeLicrHistoneEsb4H3k4me3ME0C57bl6StdPk.broadPeak",
"/oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/ENCODE/ChIP-seq/wgEncodeLicrTfbsEsb4P300ME0C57bl6StdPk.broadPeak",
"/oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/ENCODE/ChIP-seq/wgEncodeLicrTfbsEsb4CtcfME0C57bl6StdPk.broadPeak",
"/oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/ENCODE/ChIP-seq/wgEncodeLicrTfbsEsb4Pol2ME0C57bl6StdPk.broadPeak")

feat.names = c("CHD2", "HCFC1", "MAFK", "NANOG", "POU5F1", "ZC3H11A", "ZNF384",  "H3k09ac", "H3k09me3", "H3k27ac", "H3k27me3", "H3k36me3", "H3k4me1", "H3k4me3", "P300", "CTCF", "POL2")

colNum = ncol(bins)+1
for(i in 1:length(feat.names)){
	feat = read.table(file.names[i])
	colnames(feat)[1:3] = c("chr", "start", "end")
	feat.gr <- with(feat, GRanges(chr, IRanges(start+1, end), strand="*"))
	ov <- findOverlaps(bins.gr, feat.gr)
	bins[,colNum] = 0
	bins[unique(ov@queryHits), colNum] = 1
	colnames(bins)[colNum] = feat.names[i]
	colNum =  colNum + 1
}

X = t(bins[,7:ncol(bins)])
Y = promoters$FPKM

write.table(X, file = "dat_X.txt", append = FALSE, 
			quote = FALSE, sep = "\t", eol = "\n", na = "NA", 
			dec = ".", row.names = FALSE, col.names = FALSE, 
			qmethod = c("escape", "double"), fileEncoding = "")

write.table(Y, file = "dat_Y.txt", append = FALSE, 
			quote = FALSE, sep = "\t", eol = "\n", na = "NA", 
			dec = ".", row.names = FALSE, col.names = FALSE, 
			qmethod = c("escape", "double"), fileEncoding = "")

