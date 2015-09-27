library(GenomicRanges)
library(parallel)

bin_regions <- function(regions, region_len = 5000, bin_size=50){
	regions$index = 1:nrow(regions)
	bin_num = region_len/bin_size	
	regions.list = split(regions, regions$index)
	res.list <- mclapply(regions.list, function(x){
		if(x$strand=="+"){
			y = data.frame(chr=x$chr, start=seq(as.numeric(x$start), as.numeric(x$end), by=bin_size)[1:bin_num-1], end= seq(as.numeric(x$start), as.numeric(x$end), by=bin_size)[2:bin_num], index=x$index, index=1:(bin_num-1))
		}else{
			y = data.frame(chr=x$chr, start = seq(as.numeric(x$end), as.numeric(x$start), by=-bin_size)[2:bin_num], end =seq(as.numeric(x$end), as.numeric(x$start), by= -bin_size)[1:bin_num-1], index=x$index, index=1:(bin_num-1))		
		}
		return(y)
	}, mc.cores=10)
	res = do.call(rbind, res.list)
	colnames(res) = c("chr", "start", "end", "gene_id", "bin_id")
	return(res)
}

get_bin_feature <- function(bins, feat.fnames, feat.names){
	colNum = ncol(bins)+1
	bins.gr <- with(bins, GRanges(chr, IRanges(start+1, end), strand="*", gene_id, bin_id))
	for(i in 1:length(feat.names)){
		feat = read.table(feat.fnames[i])
		colnames(feat)[1:3] = c("chr", "start", "end")
		feat.gr <- with(feat, GRanges(chr, IRanges(start+1, end), strand="*"))
		ov <- findOverlaps(bins.gr, feat.gr)
		bins[,colNum] = 0
		bins[unique(ov@queryHits), colNum] = 1
		colnames(bins)[colNum] = feat.names[i]
		colNum =  colNum + 1
	}	
	bins.list <- split(bins, bins$gene_id)
	bins.array <- lapply(bins.list, function(x){t(x[,6:ncol(x)])})
	return(bins.array)
}

file_promoter="/oasis/tscc/scratch/r3fang/github/foo/results/09-25-2015/gene_promoter.bed"
file_flanking="/oasis/tscc/scratch/r3fang/github/foo/results/09-25-2015/gene_40K_flanking.bed"
file_enhancer="/oasis/tscc/scratch/r3fang/github/foo/results/09-25-2015/enhancers.2K.bed"

promoters = read.table(file_promoter)
flankings = read.table(file_flanking)
enhancers = read.table(file_enhancer)
colnames(promoters) = c("chr", "start", "end", "FPRM", "strand", "group")
colnames(flankings) = c("chr", "start", "end", "FPRM", "strand", "group")
colnames(enhancers) = c("chr", "start", "end")
enhancers$strand = "+"

promoters.gr <- with(promoters, GRanges(chr, IRanges(start, end), strand="*"))
flankings.gr <- with(flankings, GRanges(chr, IRanges(start, end), strand="*"))
enhancers.gr <- with(enhancers, GRanges(chr, IRanges(start, end), strand="*"))

ov <- findOverlaps(flankings.gr, enhancers.gr)
matches <- data.frame(promoter=ov@queryHits, enhancer=ov@subjectHits)

max_enhancer = max(table(matches[,1]))
matches = rbind(data.frame(promoter=1:nrow(promoters), enhancer=0), matches)
matches = matches[order(matches[,1]),]
#write.table(matches, file = "matches.txt", append = FALSE, quote = FALSE, sep = "\t",
#eol = "\n", na = "NA", dec = ".", row.names = FALSE, col.names = FALSE, qmethod = c("escape", "double"),
#fileEncoding = "")

bins.promoters <- bin_regions(promoters, region_len = 3000, bin_size=50)
bins.enhancers <- bin_regions(enhancers, region_len = 2000, bin_size=50)

feat.fnames = c(
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

bin.enhancers.feat <- get_bin_feature(bins.enhancers, feat.fnames, feat.names)
bin.promoters.feat <- get_bin_feature(bins.promoters, feat.fnames, feat.names)

max_enhancer=3
total_col = 39*max_enhancer+59
matches.list <- split(matches, matches[,1])

res.list <- mclapply(matches.list, function(tmp){
	p = bin.promoters.feat[[tmp[1,1]]]
	for(j in tmp[1:min(max_enhancer+1, nrow(tmp)),2]){
		if(j!=0){
			e = bin.enhancers.feat[[j]]
			p = cbind(p, e)		
		}
	}
	mm <- data.frame(matrix(0, nrow(p), total_col-ncol(p)))
	p = cbind(p, mm)
	colnames(p) = paste("X", 1:total_col, sep="")
	return(p)
}, mc.cores=10)

res = do.call(rbind, res.list)


write.table(res, file = "datX.dat", append = FALSE, quote = FALSE, sep = "\t",
eol = "\n", na = "NA", dec = ".", row.names = FALSE, col.names = FALSE, qmethod = c("escape", "double"),
fileEncoding = "")

write.table(promoters$group, file = "datY.dat", append = FALSE, quote = FALSE, sep = "\t",
eol = "\n", na = "NA", dec = ".", row.names = FALSE, col.names = FALSE, qmethod = c("escape", "double"),
fileEncoding = "")

