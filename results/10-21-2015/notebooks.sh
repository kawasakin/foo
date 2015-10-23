#PBS -q hotel 
#PBS -N cov_net
#PBS -l nodes=3:ppn=16
#PBS -l walltime=70:00:00
#PBS -o cov_net.out
#PBS -e cov_net.err
#PBS -V
#PBS -M r3fang@ucsd.edu
#PBS -m ae
#PBS -A ren-group

# iteractive nodes
### qsub -I -q hotel -N cov_net -l nodes=10:ppn=2 -l walltime=100:00:00 -A ren-group 

PYTHON=/oasis/tscc/scratch/r3fang/usr/local/python2.7.10/usr/local/bin/python2.7
GENE_FILE=/oasis/tscc/scratch/r3fang/github/foo/results/09-12-2015/mESC-zy27.gene.expr.sel
ENHANCER_FILE=/oasis/tscc/scratch/r3fang/github/foo/results/09-12-2015/mESC.enhancer.txt
DOMAINS=/oasis/tscc/scratch/r3fang/github/foo/results/10-21-2015/mESC.combined.HindIII.domain.txt

cd /oasis/tscc/scratch/r3fang/github/foo/results/10-21-2015

# 0. add index to domains
awk '{printf "%s\t%d\t%d\t%d\n", $1, $2, $3, NR}' domains.bed > domains.tmp.bed
mv domains.tmp.bed domains.bed

# 1. extract genes
/opt/R/bin/Rscript extract_genes.R 2500 2500 promoters.tmp.bed # 5k for every promoter
intersectBed -wo -f 0.90 -a promoters.tmp.bed -b domains.bed > promoters.bed 
rm promoters.tmp.bed

# 2. extreact enhancers (4k) within domains
awk '{printf "%s\t%d\t%d\n", $1, $2-500, $3+500}' /oasis/tscc/scratch/r3fang/github/foo/results/09-12-2015/mESC.enhancer.txt |\
intersectBed -wo -f 0.90 -a - -b domains.bed > enhancers.bed 

# 3. extract the enhancer-promoter accessibility
extract_accessibility.R


# 5. bin the region and extract features
# /opt/R/bin/Rscript extract_feats.R

# 6. predict the loops
# cov_net.py

# 7. extract loops from raw data
extract_loops.R 

Sox2 = 10952
SE   = 

