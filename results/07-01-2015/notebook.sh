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
###### qsub -I -q hotel -N GeneExp -l nodes=6:ppn=2 -l walltime=100:00:00 -A ren-group 

# constant
RNA_SEQ_DIR="/oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/ENCODE/RNA-seq/"
GENOME="/oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/Sequence/WholeGenomeFasta/genome.fa"
GTF="/oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/Annotation/Genes/genes.gtf"

cd /oasis/tscc/scratch/r3fang/github/foo/results/07-01-2015/

#0. delete all files under current folder with size 0
find . -size  0 -print0 | xargs -0 rm

#1. counting expression of genes counting 
files=`ls $RNA_SEQ_DIR`
for name in $files
do
	if [[ $name == *".bam" ]]
	then
		if [ ! -f "${name%.bam}.count" ]
			then
				samtools view -h $RNA_SEQ_DIR/$name | python2.7 -m HTSeq.scripts.count -m union - $GTF > "${name%.bam}.count" &
			fi
	fi
done

#2. convert count to CPM
#files=`ls .`
#for name in $files 
#do
#	if [ -s "$name" ]
#	then
#		if [[ $name == *".count" ]]
#		then
#			echo $name
#			cat $name | python ../../bin/CPM.py - > "${name%.count}.CPM"
#		fi
#	fi
#done

#3. normailize by R
#ls | grep CPM | grep -v name > CPM_name.txt 
# R
#name <- read.table('CPM_name.txt')
#data <- read.table(as.character(name[1,]))
#rownames(data) <- data[,1]
#for(i in 2:nrow(name)){
#	item <- name[i,]
#	tmp <- read.table(as.character(item))
#	data <- cbind(data, tmp[,2])
#}
#data <- data[,-1]
#a <- apply(data, 1, var)

#write.table(data.frame(a[order(a)]), file = "tmp.txt", append = FALSE, quote = FALSE, sep = "\t",
#                 eol = "\n", na = "NA", dec = ".", row.names = TRUE,
#                 col.names = FALSE, qmethod = c("escape", "double"),
#                 fileEncoding = "")