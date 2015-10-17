data = read.table("res_enh.300k.3E.rep1.txt")
data_random = read.table("res_enh.300k.3E.random.txt")
data_random
res = data.frame()
res_random = data.frame()
for(i in 0:58*15+5){
	res = rbind(res, data[i:(i+10),])
	res_random = rbind(res_random, data_random[i:(i+10),]) 
}

res_random_mean = data.frame(do.call(rbind, lapply(split(res_random[,1:2], res_random$V1), colMeans)))
res_random_min  = data.frame(do.call(rbind, lapply(split(res_random[,1:2], res_random$V1), function(x) apply(x, 2, min))))
res_random_max  = data.frame(do.call(rbind, lapply(split(res_random[,1:2], res_random$V1), function(x) apply(x, 2, max))))
res_random = data.frame(cbind(X=res_random_mean[,1], min=res_random_min[,2], mean=res_random_mean[,2], max=res_random_max[,2]))
res_random$group = "random"

res_mean = data.frame(do.call(rbind, lapply(split(res[,1:2], res$V1), colMeans)))
res_min  = data.frame(do.call(rbind, lapply(split(res[,1:2], res$V1), function(x) apply(x, 2, min))))
res_max  = data.frame(do.call(rbind, lapply(split(res[,1:2], res$V1), function(x) apply(x, 2, max))))
res      = data.frame(cbind(X=res_mean[,1], min=res_min[,2], mean=res_mean[,2], max=res_max[,2]))
res$group = "postive"
data = rbind(res, res_random)

pd <- position_dodge(0.4) # move them .05 to the left and right
ggplot(data, aes(x=X, y=mean, colour=group, group=group)) + 
    geom_errorbar(aes(ymin=min, ymax=max), colour="black", width=1.5, position=pd) +
    geom_line(position=pd) +
    geom_point(position=pd)+
	theme_bw()+
	theme(legend.position="none", axis.title.x=element_blank(),
	axis.title.y=element_blank())
	