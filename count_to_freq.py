# ---------
# Devin Suy
# ---------

import sys
import json


with open("9-gram.txt", "r") as o:
    lines = []
    for line in o:
        lines.append(line)


n_gram_counts = {}
total_count = 0

for line in lines:
    if "gram" in line: continue
    n_gram, count = line.split("\t")
    n_gram_counts[n_gram] = float(count[:-1])
    total_count += float(count)


print(total_count)

n_gram_freq = {}

for n_gram, count in n_gram_counts.items():
    n_gram_freq[n_gram] = float(count / total_count)


sorted_ngrams = [n_gram for n_gram in sorted(n_gram_freq, key=n_gram_freq.__getitem__, reverse=True)]


print(sorted_ngrams)


with open("9_gram_freq.txt", "w") as o:
    for n_gram in sorted_ngrams:
        o.write(n_gram + "," + str(n_gram_freq[n_gram]) + "\n")