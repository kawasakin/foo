import sys
import cPickle as pickle

with open(sys.argv[1]) as fin:
    promoters = pickle.load(fin)

for line in sys.stdin:
    elems = line.strip().split()
    counts = elems[9]
    pvalue = elems[10]
    target = '\t'.join(elems[6:9])
    IDs = elems[4].split('|')
    for item in IDs:
        if item in promoters:
            print target + '\t' + promoters[item] + '\t' + item + '\t' + counts + '\t' + pvalue

    