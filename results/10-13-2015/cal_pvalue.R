num_times = 59
num_times = 52 # for rep2
num_times = 59 # for rep2

matches = read.table("matches_distance.txt")
matches = matches + 1
targets = read.table("loops.300K.3E.rep3.txt")
targets$times = targets$V16*num_times
p0 = data.frame(1/table(matches[,1]))
targets$p0 = p0[match(targets$V14, p0$Var1),2]
targets$p.value = apply(targets, 1, function(x) binom.test(as.numeric(x[17]), num_times, as.numeric(x[18]), alternative="greater")$p.value) 
targets$p.value.correct = p.adjust(targets$p.value, method = "bonferroni", n = length(targets$p.value))
targets.sel <- targets[which(targets$p.value.correct<0.01),]
targets.sel$log.p.value.correct = -log10(targets.sel$p.value.correct)
targets.sel = targets.sel[order(targets.sel$log.p.value.correct, decreasing=TRUE),]
res <- targets.sel[,c(1, 3:6, 9:16, 21)]
colnames(res) <- c("gname", "gchrom", "gstart", "gend", 
                   "log.FPRM", "strand", "status", "echrom", 
				   "estart", "end", "gid", "eid", "prob", "log.p.value.correct")

write.table(res, file = "loops.rep3.sel.txt", append = FALSE, quote = FALSE, sep = "\t",
                 eol = "\n", na = "NA", dec = ".", row.names = FALSE,
                 col.names = TRUE, qmethod = c("escape", "double"),
                 fileEncoding = "")

res1 = read.table("loops.rep1.sel.txt", head=TRUE)
res2 = read.table("loops.rep2.sel.txt", head=TRUE)
res3 = read.table("loops.rep2.sel.txt", head=TRUE)
res1 = res1[which(paste(res1$gid, res1$eid, ".") %in% paste(res3$gid, res3$eid, ".")),]
res3 = res3[which(paste(res3$gid, res3$eid, ".") %in% paste(res1$gid, res1$eid, ".")),]
res1$ee = res3$log.p.value.correct[match(paste(res1$gid, res1$eid, "."), paste(res3$gid, res3$eid, "."))]


