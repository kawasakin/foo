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

# 2. obtain [-3000, +2000] region of the most & least expressed 1000 as candiates
data = read.table("mESC-zy27.gene.expr", head=T)
data.sorted = data[order(data$FPKM),]
data.sel = rbind(head(data.sorted, n=1200), tail(data.sorted, n=1200))
data.sel$label = c(rep(0, 1200), rep(1, 1200))
genes = read.table("genes.gtf")
colnames(genes) = c("chr", "strand", "gene_id")
names <- unique(intersect(data.sel$gene_id, genes$gene_id))

data.sel <- data.sel[which(data.sel$gene_id %in% names),]
genes.sel <- genes[which(genes$gene_id %in% names),]
# add strand
data.sel$strand = genes.sel$strand[match(data.sel$gene_id, genes.sel$gene_id)]
res = data.frame()
for(i in 1:nrow(data.sel)){
	x = data.sel[i,]
	if(x$strand=="-"){
		res = rbind(res, data.frame(chr=x$chr, left=x$right-2000, right=x$right+3000, strand=x$strand, FPKM=x$FPKM, gene_id=x$gene_id, label=x$label))
	}else{
		res = rbind(res, data.frame(chr=x$chr, left=x$left-3000, right=x$left+2000, strand=x$strand, FPKM=x$FPKM, gene_id=x$gene_id, label=x$label))
	}
}
write.table(res, file = "mESC_sample.txt", append = FALSE, quote = FALSE, sep = "\t",
            eol = "\n", na = "NA", dec = ".", row.names = FALSE,
            col.names = FALSE, qmethod = c("escape", "double"),
            fileEncoding = "")

# 3. Obtain the feature vector
library("GenomicRanges")
bin_size = 50
region_size = 5000
bin_num = region_size/bin_size
data = read.table("mESC_sample.txt")
data$index = 1:nrow(data)
res = data.frame()
for(i in 1:nrow(data)){
	x = data[i,]
	y = data.frame(chr=x[1], start=seq(as.numeric(x[2]), as.numeric(x[3]), bin_size)[1:bin_num-1], end= seq(as.numeric(x[2]), as.numeric(x[3]), bin_size)[2:bin_num], gene_id=x$index, index=1:(bin_num-1))
	res = rbind(res, y)
}
colnames(res) = c("chr", "start", "end", "gene_id", "bin_id")
res.gr <- with(res, GRanges(chr, IRanges(start+1, end), strand="*", gene_id, bin_id))

file.names = c(
"/oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/ENCODE/ChIP-seq/E14_CHD2.conservative.regionPeak",
"/oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/ENCODE/ChIP-seq/E14_MAFK.conservative.regionPeak",
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

feat.names = c(
"CHD2", "MAFK", "H3k09ac", "H3k09me3", "H3k27ac", "H3k27me3", "H3k36me3", "H3k4me1", "H3k4me3", "P300", "CTCF", "POL2")

colNum = ncol(res)+1
for(i in 2:length(feat.names)){
	feat = read.table(file.names[i])
	colnames(feat)[1:3] = c("chr", "start", "end")
	feat.gr <- with(feat, GRanges(chr, IRanges(start+1, end), strand="*"))
	ov <- findOverlaps(res.gr, feat.gr)
	res[,colNum] = 0
	res[unique(ov@queryHits), colNum] = 1
	colnames(res)[colNum] = feat.names[i]
	colNum =  colNum + 1
}

res.list = split(res, res$gene_id)
res.array <- lapply(res.list, function(x){
	rapply(x[,6:ncol(x)], c)
})

dat = data.frame(t(data.frame(res.array)))
dat$label = data$V7

write.table(dat, file = "mESC_sample_features.txt", append = FALSE, quote = FALSE, sep = "\t",
                eol = "\n", na = "NA", dec = ".", row.names = FALSE,
                col.names = FALSE, qmethod = c("escape", "double"),
                fileEncoding = "")

# 4. predict by logisitic regression
library(lars)
dat = read.table("mESC_sample_features.txt")
colnames(dat)[ncol(dat)] = "label"
index.train = sample(1:nrow(dat), 4*nrow(dat)/5)
dat.train = dat[index.train,]
dat.test = dat[-index.train,]

logr_vm <- glm(label ~ ., data=dat.train, family=binomial(link="logit"))
order(logr_vm$coefficients, decreasing=TRUE)

pred <- predict(logr_vm, dat.test, type="response")

pred[which(pred>0.6)]=1
pred[which(pred<=0.6)]=0
length(which(pred==dat.test$label))




