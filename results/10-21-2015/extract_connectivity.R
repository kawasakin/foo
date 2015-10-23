library(GenomicRanges)
access = read.table("access.txt")
promoters = read.table("promoters.bed")
enhancers = read.table("enhancers.bed")
pairs = cbind(promoters[access[,1],1:6], enhancers[access[,2],1:3], access)

colnames(pairs) = c("gchrom", "gstart", "gend", "FPRM", "strand", "group", "echrom", "estart", "end", "gid", "eid")

loops1  = read.table("/oasis/tscc/scratch/r3fang/data/Mus_musculus/Hi-C/P_Fraser/mESC/rep1/ERR466159.filtered.merged.sorted.cis.bed")
loops2  = read.table("/oasis/tscc/scratch/r3fang/data/Mus_musculus/Hi-C/P_Fraser/mESC/rep1/ERR466164.filtered.merged.sorted.cis.bed")
loops3  = read.table("/oasis/tscc/scratch/r3fang/data/Mus_musculus/Hi-C/P_Fraser/mESC/rep1/ERR466169.filtered.merged.sorted.cis.bed")


#loops2  = read.table("rep2.sorted.filtered.cis.100K.bed")
#loops = rbind(loop1, loops2)
loops = loops1

loops_left.gr  <- GRanges(seqnames=loops[,2], ranges=IRanges(loops[,3]-2000, loops[,3]+2000), strand="*")
loops_right.gr  <- GRanges(seqnames=loops[,2], ranges=IRanges(loops[,4]-2000, loops[,4]+2000), strand="*")

pair1 = pairs[which(pairs$gstart-pairs$estart<0),]
pair2 = pairs[which(pairs$gstart-pairs$estart>=0),]

countHiReads_pair1 <- function(pair1, loops_left.gr, loops_right.gr){
	pair1_left.gr  <- GRanges(seqnames=pair1$gchrom, ranges=IRanges(pair1$gstart, pair1$gend), strand="*")
	pair1_right.gr <- GRanges(seqnames=pair1$echrom, ranges=IRanges(pair1$estart, pair1$end), strand="*")

	ov_left = data.frame(findOverlaps(pair1_left.gr, loops_left.gr))
	ov_right = data.frame(findOverlaps(pair1_right.gr, loops_right.gr))

	ov_left$name = paste(ov_left[,1], ov_left[,2], sep=".")
	ov_right$name = paste(ov_right[,1], ov_right[,2], sep=".")

	tmp = ov_left[which(ov_left$name %in% ov_right$name),]
	freq1 = data.frame(table(tmp$queryHits))

	pair1$HiReadNum=0
	pair1$HiReadNum[as.numeric(levels(freq1$Var1))[freq1$Var1]] = freq1$Freq
	return(pair1)
}
countHiReads_pair2 <- function(pair1, loops_left.gr, loops_right.gr){
	pair1_right.gr  <- GRanges(seqnames=pair1$gchrom, ranges=IRanges(pair1$gstart, pair1$gend), strand="*")
	pair1_left.gr <- GRanges(seqnames=pair1$echrom, ranges=IRanges(pair1$estart, pair1$end), strand="*")

	ov_left = data.frame(findOverlaps(pair1_left.gr, loops_left.gr))
	ov_right = data.frame(findOverlaps(pair1_right.gr, loops_right.gr))

	ov_left$name = paste(ov_left[,1], ov_left[,2], sep=".")
	ov_right$name = paste(ov_right[,1], ov_right[,2], sep=".")

	tmp = ov_left[which(ov_left$name %in% ov_right$name),]
	freq1 = data.frame(table(tmp$queryHits))

	pair1$HiReadNum=0
	pair1$HiReadNum[as.numeric(levels(freq1$Var1))[freq1$Var1]] = freq1$Freq
	return(pair1)
}

res.pair1 <- countHiReads_pair1(pair1, loops_left.gr, loops_right.gr)
res.pair2 <- countHiReads_pair2(pair2, loops_left.gr, loops_right.gr)
res <- rbind(res.pair1, res.pair2)

res <- res[order(res$log.p.value.correct, decreasing=TRUE),]
res$len = res$gend - res$gstart + res$end - res$estart
res$HiReadNum.norm = 1000*res$HiReadNum/res$len

pairs = read.table("loops.rep1.sel.txt", head=TRUE)
pairs$dist = apply(data.frame(abs(pairs$gstart - pairs$estart), abs(pairs$gstart - pairs$end), abs(pairs$gend - pairs$end), abs(pairs$gend - pairs$estart)), 1, min)
length(which(paste(pairs$gid, pairs$eid, sep=".") %in% paste(loops$V7, loops$V11, sep=".")))
pairs.sel = pairs[which(paste(pairs$gid, pairs$eid, sep=".") %in% paste(loops$V7, loops$V11, sep=".")),]

