import collections 
import sys

def get_exp_interval(genes, exp_num, bin_size, upstream, downstream):
    exps = [x[3] for x in genes]
    exp_interval = [exps[x] for x in range(0, len(exps), len(exps)/exp_num)]
    exp_interval.append([max(exps)])
    for [chrom, left, right, exp, strand, name, index] in genes:
        if(exp >= max(exp_interval)):
            group = exp_pattern_num
        else:
            group = [exp<x for x in exp_interval].index(True)
        if(strand=="+"):
            right = left + downstream
            left = left - upstream
            for i in range(left, right, bin_size):
                print chrom, i+1, i+bin_size, exp, name, group, index
        else:
            left = right - downstream
            right = right + upstream
            for i in range(left, right, bin_size)[::-1]:
                print chrom, i+1, i+bin_size, exp, name, group, index
        
        
def read_gene(fname):
    genes = []
    j = 0
    with open(fname) as fin:
        for line in fin:
            j = j + 1
            [gene_id, num, chrom, left, right, exp, min_exp, max_exp, strand] = line.strip().split()
            genes.append([chrom, int(left), int(left), float(exp), strand, gene_id, j])
    return genes

def main():
    fname = "mESC-zy27.gene.expr.sel"
    gene_num = 4000
    upstream = 3000
    downstream = 2000
    bin_size = 50
    exp_pattern_num = 2
    bin_num = (downstream+upstream)/bin_size
    genes = read_gene(fname)
    genes.sort(key=lambda x: x[3])
    genes_sel = genes[0:gene_num/2] + genes[len(genes)-gene_num/2:]
    get_exp_interval(genes_sel, exp_pattern_num, bin_size, upstream, downstream)

if __name__ == "__main__":
    main()
