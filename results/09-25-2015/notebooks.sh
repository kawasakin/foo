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
GENE_FILE=/oasis/tscc/scratch/r3fang/github/foo/results/09-12-2015/mESC-zy27.gene.expr.sel
ENHANCER_FILE=/oasis/tscc/scratch/r3fang/github/foo/results/09-12-2015/mESC.enhancer.txt

cd /oasis/tscc/scratch/r3fang/github/foo/results/09-25-2015
# 1. extract studied genes
#/opt/R/bin/Rscript extract_genes.R 20000 20000 gene_40K_flanking.bed

# 2. extract promoter regions
#/opt/R/bin/Rscript extract_genes.R 3000 0 gene_promoter.bed

# 3. extreact enhancers that overlap with flanking regions
#awk '{printf "%s\t%d\t%d\n", $1, $2+1000, $3-1000}' $ENHANCER_FILE |\
#intersectBed -u -wa -a - -b gene_40K_flanking.bed | sort - | uniq - > enhancers.2K.bed 

# 4. count how many enhancers within 40kb of selected genes
#intersectBed -c -wa -a gene_40K_flanking.bed -b enhancers.2K.bed > tmp.txt

# 5. bin the region and extract features
/opt/R/bin/Rscript extract_feats.R
