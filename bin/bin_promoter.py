import collections 

def main():
    gtf = {}
    with open("genes.gtf") as fin:
        for line in fin:
            