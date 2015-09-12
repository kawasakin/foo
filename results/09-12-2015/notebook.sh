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

# 0. download gene expression data
wget http://chromosome.sdsc.edu/mouse/download/19-tissues-expr.zip 
unzip 19-tissues-expr.zip 
