import sys
import collections

counts_sum = 0;
counts = collections.defaultdict(float)
 
for line in sys.stdin:
    if line.startswith('__'):
        counts_sum += int(line.strip().split()[1])
    else:
        [name, num] = line.strip().split()
        counts[name] = float(num)

for name in counts:
    print name, (counts[name]/counts_sum)*1000000