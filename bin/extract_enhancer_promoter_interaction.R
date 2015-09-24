library(GenomicRanges)
library(parallel)

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

bin_regions <- function(regions, region_len = 5000, bin_size=50){
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

get_gene_enhancer_match <- function(genes, loops, enhancers){
	genes.gr <- with(genes, GRanges(chr, IRanges(start+1, end), strand="*", index))
	loops.promoter.gr <- with(loops, GRanges(chr.bait, IRanges(start.bait+1, end.bait), strand="*", index))
	loops.enhancer.gr <- with(loops, GRanges(chr, IRanges(start+1, end), strand="*", index))
	enhancers.gr <- with(enhancers, GRanges(chr, IRanges(start+1, end), strand="*", index))

	ov1 <- findOverlaps(genes.gr, loops.promoter.gr)
	ov2 <- findOverlaps(loops.promoter.gr, enhancers.gr)

	matches <- ov2@subjectHits[match(ov1@subjectHits, ov2@queryHits)]
	matches[which(is.na(matches))] = 0 
	matches <- unique(data.frame(promoters = ov1@queryHits, enhancers=matches))
	matches <- matches[which(matches$enhancers>0),]
	return(matches)
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
	res <- data.frame(do.call(rbind, bins.array))	
	return(res)
}

file_gene = "mESC-zy27.gene.expr.sel"
file_enhancer = "mESC.enhancer.txt"
file_loop1 = "/oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/CHIC/ESC_promoter_promoter_significant_interactions.txt"
file_loop2 = "/oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/CHIC/ESC_promoter_other_significant_interactions.txt"

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


loops1 = read.table(file_loop1, sep="\t", head=T)
loops2 = read.table(file_loop2, sep="\t", head=T)

loops1 = loops1[,match(c("chr", "start", "end", "chr.1", "start.1", "end.1"), colnames(loops1))]
loops2 = loops2[,match(c("chr.bait", "start.bait", "end.bait", "chr", "start", "end"), colnames(loops2))]
colnames(loops1) = c("chr.bait", "start.bait", "end.bait", "chr", "start", "end")
loops = rbind(loops1, loops2)
loops$index = 1:nrow(loops)

genes = read.table(file_gene, head=T)
genes$FPKM = log(genes$FPKM)
genes = genes[order(genes$FPKM),]
genes = rbind(genes[1:4431,], genes[(nrow(genes)-6609):nrow(genes),])
genes$label = c(rep(0, 4431), rep(1, 6610))

genes = genes[,c(3, 4, 5, 6, 9, 10)]
colnames(genes) = c("chr", "start", "end", "FPKM", "strand", "label")
genes = genes[order(genes$FPKM),]
genes$index = 1:nrow(genes)

enhancers = read.table(file_enhancer, head=T)
enhancers$strand = "+"
enhancers$index = 1:nrow(enhancers)

## extending genes' region 
genes.extended = genes
genes.extended$start = genes.extended$start - 3000
genes.extended$end = genes.extended$end + 3000

matches <- get_gene_enhancer_match(genes.extended, loops, enhancers)

# filter enhancers not overlaped
enhancers.sel <- enhancers[unique(matches$enhancers),]
matches <- get_gene_enhancer_match(genes.extended, loops, enhancers.sel)

promoters <- get_promoter(genes, 3000, 2000)
bins.enhancers <- bin_regions(enhancers.sel, region_len = 4000, bin_size=50)
bins.promoters <- bin_regions(promoters, region_len = 5000, bin_size=50)

colnames(bins.enhancers) = c("chr", "start", "end", "gene_id", "bin_id")
colnames(bins.promoters) = c("chr", "start", "end", "gene_id", "bin_id")


a <- get_bin_feature(bins.enhancers, feat.fnames, feat.names)
b <- get_bin_feature(bins.promoters, feat.fnames, feat.names)


write.table(enhancers.sel, file = "enhancer.sel.txt", append = FALSE, quote = FALSE, sep = "\t",
                 eol = "\n", na = "NA", dec = ".", row.names = FALSE,
                 col.names = FALSE, qmethod = c("escape", "double"),
                 fileEncoding = "")

write.table(a, file = "dat_X_E.txt", append = FALSE, quote = FALSE, sep = "\t",
                 eol = "\n", na = "NA", dec = ".", row.names = FALSE,
                 col.names = FALSE, qmethod = c("escape", "double"),
                 fileEncoding = "")

write.table(b, file = "dat_X_P.txt", append = FALSE, quote = FALSE, sep = "\t",
                 eol = "\n", na = "NA", dec = ".", row.names = FALSE,
		         col.names = FALSE, qmethod = c("escape", "double"),
				 fileEncoding = "")

write.table(genes$label, file = "dat_Y.txt", append = FALSE, quote = FALSE, sep = "\t",
                 eol = "\n", na = "NA", dec = ".", row.names = FALSE,
		         col.names = FALSE, qmethod = c("escape", "double"),
				 fileEncoding = "")


write.table(matches, file = "interaction.txt", append = FALSE, quote = FALSE, sep = "\t",
			eol = "\n", na = "NA", dec = ".", row.names = FALSE,
			col.names = FALSE, qmethod = c("escape", "double"),
			fileEncoding = "")
