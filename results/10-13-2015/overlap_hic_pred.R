loops = read.table("loops_HiC.txt")
colnames(loops) = c("gchrom", "gstart", "gend", "FRPM", "strand", "group", "gid", "echrom", "estart", "end", "eid", "estrand")
preds = read.table("loops.rep3.sel.txt", head=TRUE)
preds = preds[which(preds$prob>0.9),]
preds$dist = apply(data.frame(abs(preds$gstart - preds$estart), abs(preds$gstart - preds$end), abs(preds$gend - preds$end), abs(preds$gend - preds$estart)), 1, min)
loops$dist = apply(data.frame(abs(loops$gstart - loops$estart), abs(loops$gstart - loops$end), abs(loops$gend - loops$end), abs(loops$gend - loops$estart)), 1, min)

preds.sel = preds[which(preds$dist>=100000),]
loops.sel = loops[which(loops$dist>=100000 & loops$dist <= 150000),]

preds.sel.names = paste(preds.sel$gid, preds.sel$eid, sep = ".")
loops.sel.names = paste(loops.sel$gid, loops.sel$eid, sep = ".")

length(which(preds.sel.names %in% loops.sel.names))/length(preds.sel.names)
length(which(preds.sel.names %in% loops.sel.names))/length(loops.sel.names)

