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

cd /oasis/tscc/scratch/r3fang/github/foo/results/07-01-2015/

#0. delete all files under current folder with size 0
# find . -size  0 -print0 | xargs -0 rm

#1. counting expression of genes counting 
# files=`ls $RNA_SEQ_DIR`
# for name in $files
# do
# 	if [[ $name == *".bam" ]]
# 	then
# 		if [ ! -f "${name%.bam}.count" ]
# 			then
# 				samtools view -h $RNA_SEQ_DIR/$name | python2.7 -m HTSeq.scripts.count -m union - $GTF > "${name%.bam}.count" &
# 			fi
# 	fi
# done

#2. convert count to CPM
# files=`ls .`
# for name in $files 
# do
# 	if [ -s "$name" ]
# 	then
# 		if [[ $name == *".count" ]]
# 		then
# 			echo $name
# 			cat $name | python ../../bin/CPM.py - > "${name%.count}.CPM"
# 		fi
# 	fi
# done

#3. select a subset of genes with mean > 5 and sd > 10 to study 
# get the name of all .CPM files
# ls | grep CPM | grep -v name > CPM_name.txt 

# R
 library(Hmisc)
 name <- read.table('CPM_name.txt')
 data <- read.table(as.character(name[1,]))
 for(i in 2:nrow(name)){
 	item <- name[i,]
 	tmp <- read.table(as.character(item))
 	data <- cbind(data, tmp[,2])
 }
 rownames(data) <- data[,1]
 data <- data[,-1]
 # calculate mean and standard variance for each gene
 sd_rows <- apply(data, 1, sd)
 mean_rows <- apply(data, 1, mean)

 sd_cols <- apply(data, 2, sd)
 mean_cols <- apply(data, 2, mean)
 
 sd_rows[which(is.na(sd_rows))] = 0
 mean_rows[which(is.na(mean_rows))] = 0
 data_filtered <- data[intersect(which(sd_rows>10), which(mean_rows>10)), which(mean_cols>10)]
 
 # hcluster 
 data_filtered = t(as.matrix(data_filtered)) 
 corRaw = cor(data_filtered, method="pearson")
 dissimilarity <- 1 - corRaw
 distance <- as.dist(dissimilarity)
 hc <- hclust(distance)
 clusters <- cutree(hc, h = 0.6)
 data_filtered <- data_filtered[,order(clusters)]
 clusters <- clusters[order(clusters)]
 
 clusters.sel <- clusters[clusters %in% as.numeric(which(table(clusters)>50))]
 data.sel <- data_filtered[,which(colnames(data_filtered)%in%names(clusters.sel))]
 
 library(spatstat)
 corRaw <- cor(data.sel)
 plot(im(corRaw[nrow(corRaw):1,]), main="Correlation Matrix Map")
 
 rm(corRaw)
 rm(dissimilarity)
 rm(distance)
 rm(hc)
 
 save.images("gene_cluster.RData")
 
 data.sel <- t(data.sel)
 data.sel.scale <- t(apply(data.sel, 1, function(x) (x-min(x))/(max(x)-min(x))))
 plot(im(data.sel.scale[nrow(data.sel.scale):1,]))
 
 heatmap(data.sel.scale, col=topo.colors(100))
