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

### qsub -I -q hotel -N cov_net -l nodes=10:ppn=2 -l walltime=100:00:00 -A ren-group 
### qsub -I -q pdafm -N cov_net -l nodes=32:ppn=3 -l walltime=72:00:00 -A ren-group 

### qsub -I -q gpu-hotel -N ConvNet-GPU -l nodes=1:ppn=3 -l walltime=100:00:00 -A ren-group
PYTHON=/oasis/tscc/scratch/r3fang/usr/local/python2.7.10/usr/local/bin/python2.7
GENE_FILE=/oasis/tscc/scratch/r3fang/github/foo/results/09-12-2015/mESC-zy27.gene.expr.sel
ENHANCER_FILE=/oasis/tscc/scratch/r3fang/github/foo/results/09-12-2015/mESC.enhancer.txt

cd /oasis/tscc/scratch/r3fang/github/foo/results/09-25-2015

# 1. extract studied genes
/opt/R/bin/Rscript extract_genes.R 2500 2500 promoters.bed 

# 2. extract promoter regions
/opt/R/bin/Rscript extract_genes.R 200000 200000 flankings.bed 

# 3. extreact enhancers that overlap with flanking regions
awk '{printf "%s\t%d\t%d\n", $1, $2+1500, $3-1500}' /oasis/tscc/scratch/r3fang/github/foo/results/09-12-2015/mESC.enhancer.txt |\
intersectBed -u -wa -a - -b flankings.bed | sort - | uniq - > enhancers.bed 

# 4. extract the enhancer-promoter matches by distance and Hi-C
extract_loops.R 

# 5. bin the region and extract features
# /opt/R/bin/Rscript extract_feats.R

# 6. predict the loops
# cov_net.py

# 7. extract loops from raw data
extract_loops.R 

Sox2 = 10952
SE   = 

