# draw heatmap
data = read.table("feat_heatmaps.txt")
t = 9
x = data[(11041*t+1):(11041*(t+1)),]
xmin = 0
xmax = 4
collist<-c("#053061","#2166AC","#4393C3","#92C5DE","#D1E5F0","#F7F7F7","#FDDBC7","#F4A582","#D6604D","#B2182B","#67001F")
ColorRamp<-colorRampPalette(collist)(10000)
ColorLevels<-seq(from=xmin, to=xmax, length=10000)
ColorRamp_ex <- ColorRamp[ColorLevels]
ColorRamp_ex <- ColorRamp[round(1+(min(x)-xmin)*10000/(xmax-xmin)) : round( (max(x)-xmin)*10000/(xmax-xmin) )]
image(t(as.matrix(x)), col=ColorRamp_ex, las=1, xlab="",ylab="",cex.axis=1,xaxt="n",yaxt="n")


motifs = read.table("weights.txt")
t = 9
x = motifs[(17*t+1):(17*(t+1)),]
orders = c(1, 16, 8, 3, 10, 17, 6, 13, 14, 11, 15, 2, 4, 9, 12, 5, 7)
x = x[orders,]
xmin = min(x)
xmax = max(x)
collist<-c("darkblue", "yellow")
ColorRamp<-colorRampPalette(collist)(3000)
ColorLevels<-seq(from=xmin, to=xmax, length=3000)
ColorRamp_ex <- ColorRamp[round(1+(min(x)-xmin)*3000/(xmax-xmin)) : round( (max(x)-xmin)*3000/(xmax-xmin) )]
image(t(as.matrix(x)), col=ColorRamp_ex, las=1, xlab="",ylab="",cex.axis=1,xaxt="n",yaxt="n")

feat.names = c("CHD2", "HCFC1", "MAFK", "NANOG", "POU5F1", "ZC3H11A", "ZNF384",  "H3k09ac", "H3k09me3", "H3k27ac", "H3k27me3", "H3k36me3", "H3k4me1", "H3k4me3", "P300", "CTCF", "POL2")
rev(feat.names)[orders]

