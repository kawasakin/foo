library(GenomicRanges)
library(parallel)

get_gene_enhancer_match <- function(genes, loops, enhancers){
	genes.gr <- with(genes, GRanges(chr, IRanges(start+1, end), strand="*", index))
	loops.promoter.gr <- with(loops, GRanges(chr.bait, IRanges(start.bait+1, end.bait), strand="*", index))
	loops.enhancer.gr <- with(loops, GRanges(chr, IRanges(start+1, end), strand="*", index))
	enhancers.gr <- with(enhancers, GRanges(chr, IRanges(start+1, end), strand="*", index))

	ov1 <- findOverlaps(genes.gr, loops.promoter.gr)
	ov2 <- findOverlaps(loops.enhancer.gr, enhancers.gr)
	
	matches_enhancers <- ov2@subjectHits[match(ov1@subjectHits, ov2@queryHits)]
	matches_loops <- ov2@queryHits[match(ov1@subjectHits, ov2@queryHits)]
	
	matches_enhancers[which(is.na(matches_enhancers))] = 0
	matches_loops[which(is.na(matches_loops))] = 0
	
	matches <- unique(data.frame(promoters = ov1@queryHits, enhancers=matches_enhancers, loops = matches_loops))
	matches <- matches[which(matches$enhancers>0),]
	return(matches)
}


file_promoter = "promoters.bed"
file_enhancer = "enhancers.bed"
file_loop = "/oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/CHIC/ESC_promoter_other_significant_interactions.txt"

promoters = read.table(file_promoter)
enhancers = read.table(file_enhancer)
loops     = read.table(file_loop, sep="\t", head=T)

promoters = promoters[,c(1:6, 10)]
enhancers = enhancers[,c(1:3, 7)]
colnames(promoters) = c("chr", "start", "end", "FPRM", "strand", "group", "domain")
colnames(enhancers) = c("chr", "start", "end", "domain")

loops$index = 1:nrow(loops)
enhancers$index = 1:nrow(enhancers)
promoters$index = 1:nrow(promoters)

matches <- get_gene_enhancer_match(promoters, loops, enhancers)

res = data.frame()
for(i in 1:nrow(promoters)){
	enh.sel <- which(enhancers$domain==promoters$domain[i])
	if(length(enh.sel) > 0){
		res = rbind(res, data.frame(pro = i, enh = enh.sel))		
	}
}
res$raw.count = 0
res$log.observed.expected. = 0
res$loop = 0
a = data.frame(res= 1:nrow(res), loops=match(paste(res[,1], res[,2], sep="."), paste(matches[,1], matches[,2], sep=".")))
a = a[which(!is.na(a[,2])),]
res$loop[a[,1]] = matches$loops[a[,2]]
res$raw.count[a[,1]] = loops$raw.count[matches$loops[a[,2]]]
res$log.observed.expected.[a[,1]] = loops$log.observed.expected.[matches$loops[a[,2]]]
colnames(res) = c("promoter", "enhancer", "raw.count", "log.observed.expected")

write.table(res[,1:4], file = "access.txt", append = FALSE, quote = FALSE, sep = "\t",
eol = "\n", na = "NA", dec = ".", row.names = FALSE, col.names = FALSE, qmethod = c("escape", "double"),
fileEncoding = "")
