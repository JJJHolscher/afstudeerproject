from collections import defaultdict
import numpy as np

T_START = 1810
T_END = 2010

IN_FILES = 'corpus.txt', 'shuffled_corpus.txt'
OUT_FILES = 'frequencies.txt', 'shuffled_frequencies.txt'

counter = defaultdict(lambda: np.zeros(T_END - T_START, dtype=int))

for filename in IN_FILES:
    with open(filename) as in_file:
        for row in in_file:
            row = row[:-1].split(' ')
            year, row = int(row[0].split('\t')[0]), row[1:]
            for word in row:
                counter[word][year - T_START] += 1

for filename in OUT_FILES:
    with open(filename) as out_file:
        for word, array in counter.items():
            out_file.write(word + ' ' + ' '.join([str(i) for i in array]) + '\n')
