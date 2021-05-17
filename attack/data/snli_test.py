import os
import collections
dataset = []

current_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(
    current_dir, 'snli'
)
label2id = {'entailment': 1,
            'neutral': 2,
            'contradiction': 0}   # consist with hugging face datasets

with open(file_path, 'r', encoding='utf8') as fin:
    for line in fin.readlines():
        label, premise, hypothesis = line.strip().split('\t')
        inputs = collections.OrderedDict(
            [('premise', premise),
             ('hypothesis', hypothesis)]
        )
        dataset.append((inputs, label2id[label]))
