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
file_flanking = "flankings.bed"
file_enhancer = "enhancers.bed"
file_loop     = "/oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/CHIC/ESC_promoter_other_significant_interactions.txt"

promoters = read.table(file_promoter)
flankings = read.table(file_flanking)
enhancers = read.table(file_enhancer)
loops     = read.table(file_loop, sep="\t", head=T)

colnames(promoters) = c("chr", "start", "end", "FPRM", "strand", "group")
colnames(flankings) = c("chr", "start", "end", "FPRM", "strand", "group")
colnames(enhancers) = c("chr", "start", "end")
loops$index = 1:nrow(loops)
enhancers$index = 1:nrow(enhancers)
promoters$index = 1:nrow(promoters)
enhancers$strand = "+"

promoters.gr <- with(promoters, GRanges(chr, IRanges(start, end), strand="*"))
flankings.gr <- with(flankings, GRanges(chr, IRanges(start, end), strand="*"))
enhancers.gr <- with(enhancers, GRanges(chr, IRanges(start, end), strand="*"))

ov <- findOverlaps(flankings.gr, enhancers.gr)
matches <- data.frame(promoter=ov@queryHits, enhancer=ov@subjectHits)
matches = matches[order(matches[,1]),]
write.table(matches, file = "matches_distance.txt", append = FALSE, quote = FALSE, sep = "\t",
eol = "\n", na = "NA", dec = ".", row.names = FALSE, col.names = FALSE, qmethod = c("escape", "double"),
fileEncoding = "")

matches <- get_gene_enhancer_match(promoters, loops, enhancers)
matches = matches[order(matches[,1]),]
res <- cbind(promoters[matches$promoters,], enhancers[matches$enhancers,], loops[matches$loops, c(10, 11)])

write.table(res, file = "loops_HiC.txt", append = FALSE, quote = FALSE, sep = "\t",
eol = "\n", na = "NA", dec = ".", row.names = FALSE, col.names = FALSE, qmethod = c("escape", "double"),
fileEncoding = "")
