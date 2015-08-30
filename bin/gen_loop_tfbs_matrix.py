#!~/usr/local/bin/python2.7
import sys
import collections

TFBSs = []
promoters = collections.defaultdict(list)
enhancers = collections.defaultdict(list)
loops = {}

# read loops
for line in sys.stdin:
    elems = line.strip().split()
    loops[elems[0]] = elems[1]

# read promoters
with open(sys.argv[2]) as fin:
    for line in fin:
        elems = line.strip().split()
        promoters[elems[0]].append('P.'+elems[1])
        if 'P.'+elems[1] not in TFBSs:
            TFBSs.append('P.'+elems[1])

# read enhancers
with open(sys.argv[3]) as fin:
    for line in fin:
        elems = line.strip().split()
        enhancers[elems[0]].append('E.'+elems[1])
        if 'E.'+elems[1] not in TFBSs:
            TFBSs.append('E.'+elems[1])

loops_name = list(loops.keys())

print '\t'.join(['loop_name', 'intensity'] + TFBSs)
for i in xrange(len(loops_name)):
    item = loops_name[i]
    [p, e] = item.split('_')
    a = [0]*len(TFBSs)
    for j in map(lambda x:TFBSs.index(x), enhancers[e]) + map(lambda x:TFBSs.index(x), promoters[p]):
        a[j] = 1
    print '\t'.join([item, loops[item]] + map(str, a))





