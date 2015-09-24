#PBS -q hotel 
#PBS -N convolutional_net
#PBS -l nodes=5:ppn=4
#PBS -l walltime=70:00:00
#PBS -o convolutional_net.out
#PBS -e convolutional_net.err
#PBS -V
#PBS -M r3fang@ucsd.edu
#PBS -m ae
#PBS -A ren-group
### qsub -I -q hotel -N GeneExp -l nodes=1:ppn=1 -l walltime=100:00:00 -A ren-group 
PYTHON=/oasis/tscc/scratch/r3fang/usr/local/python2.7.10/usr/local/bin/python2.7
# 1. go over ../../bin/extract_enhancer_promoter_interaction.R and generate 
# Enhancer_promoter_matches.txt
# Enhancers.bed
# teX.bed
# teX_chip.dat
# teX_Enhancer.txt
# teY.dat
# trX.bed
# trY.dat

# 2. get genomic sequence

# fastaFromBed -fi /oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/Sequence/WholeGenomeFasta/genome.fa -bed Enhancers.bed -fo Enhancers.fa
# fastaFromBed -fi /oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/Sequence/WholeGenomeFasta/genome.fa -bed trX.bed -fo trX.fa
# fastaFromBed -fi /oasis/tscc/scratch/r3fang/data/Mus_musculus/UCSC/mm9/Sequence/WholeGenomeFasta/genome.fa -bed teX.bed -fo teX.fa

cd /oasis/tscc/scratch/r3fang/github/foo/results/09-23-2015
$PYTHON convolutional_net.py > conv_net_eval.txt
