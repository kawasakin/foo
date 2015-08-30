import sys
import collections
import cPickle as pickle

genes = {}

for line in sys.stdin:
    elems = line.strip().split()
    chrom = elems[0]
    start = int(elems[1])
    end = int(elems[2])
    strand = elems[3]
    ID = elems[5].split('.')[0].replace('"', '')    
    if strand == '+':
        end = str(start)
        start = str(start - 2000)
        pos = '\t'.join([chrom, start, end, strand])
    else:
        start = str(end)
        end = str(end + 2000)
        pos = '\t'.join([chrom, start, end, strand])        
    genes[ID] = pos

with open(sys.argv[2], 'w') as fout:
    pickle.dump(genes, fout)     