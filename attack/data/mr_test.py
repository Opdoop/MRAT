import os

dataset = []

current_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(
    current_dir, 'mr'
)
with open(file_path, 'r', encoding='utf8') as fin:
    for line in fin.readlines():
        label, string = int(line[0]), line[2:].strip()
        dataset.append((string, label))
