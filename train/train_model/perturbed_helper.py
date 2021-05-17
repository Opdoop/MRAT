import csv
from .train_args_helpers import clean_str
from torch.utils.data import Dataset
import numpy as np

def batch_encode(tokenizer, text_list):
    if hasattr(tokenizer, "batch_encode"):
        return tokenizer.batch_encode(text_list)
    else:
        return [tokenizer.encode(text_input) for text_input in text_list]

def _batch_encoder(tokenizer, text):
    '''
    Large text list cause process killed. Orderly process
    :param tokenizer:
    :param text:
    :return:
    '''
    text_ids = []
    batch_number = len(text)//10000
    start, end = 0, 0
    for i in range(batch_number):
        start = i * 10000
        end = (i+1) * 10000
        text_ids.extend(batch_encode(tokenizer, text[start:end]))
    text_ids.extend(batch_encode(tokenizer, text[end:]))
    return text_ids

class PerturbedDataset(Dataset):
    def __init__(self, file_paths, tokenizer):
        self.tokenizer = tokenizer
        self.file_paths = file_paths
        self.text_list, self.perturbed_list, self.label_list = self.perturbed_dataset()

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, idx):
        return self.text_list[idx], self.perturbed_list[idx], self.label_list[idx]

    def _text_formation(self, text):
        if text.find(">>>>") > 0 :
            premise, hypothesis = text.split(">>>>")
            premise = premise[9:]
            hypothesis = hypothesis[12:]
            text = (premise, hypothesis)
        return text

    def read_csv(self, file_path, clean=False):
        '''
        read TextAttack csv result，return four list: ground_truth_output, original_text, perturbed_text, result_type
        result csv headline：
        "ground_truth_output","num_queries","original_output","original_score","original_text",\
        "perturbed_output","perturbed_score","perturbed_text","result_type"
        :param file_path:
        :return:
        '''
        with open(file_path, 'r', encoding='utf8') as fin:
            reader = csv.reader(fin)
            next(reader) # skip headline
            ground_list, original_list, perturbed_list, result_list = [], [], [], []
            for line in reader:
                ground_truth_output, _, _, _, original_text, _, _, perturbed_text, result_type = line
                if clean:
                    original_text = clean_str(original_text)
                    perturbed_text = clean_str(perturbed_text)
                ground_list.append(float(ground_truth_output))
                original_list.append(self._text_formation(original_text))
                perturbed_list.append(self._text_formation(perturbed_text))
                result_list.append(result_type)
        return ground_list, original_list, perturbed_list, result_list

    def read_perturbed_text(self, file_path):
        '''
        :param file_paths:
        :return: dataset
        '''
        ground_list, original_list, perturbed_list, result_list = self.read_csv(file_path)
        orginal_idx = _batch_encoder(self.tokenizer, original_list)
        perturbed_idx = _batch_encoder(self.tokenizer, perturbed_list)

        text_list, perturbed_list, label_list, idx_list = [], [], [], []
        for idx, result in enumerate(result_list):
            if result == 'Successful':
                text_list.append(orginal_idx[idx])
                perturbed_list.append(perturbed_idx[idx])
                label_list.append(int(ground_list[idx]))
                idx_list.append(idx)
        text_list = np.array(text_list)
        perturbed_list = np.array(perturbed_list)
        label_list = np.array(label_list)
        return text_list, perturbed_list, label_list

    def perturbed_dataset(self):
        text_all, perturbed_all, label_all = [], [], []
        for path in self.file_paths:
            text_list, perturbed_list, label_list = self.read_perturbed_text(path)
            text_all.extend(text_list)
            perturbed_all.extend(perturbed_list)
            label_all.extend(label_list)
        return text_all, perturbed_all, label_all

    def perturbed_string(self):
        perturbed_all, label_all = [], []
        for path in self.file_paths:
            ground_list, original_list, perturbed_list, result_list = self.read_csv(path)
            for i in range(len(result_list)):
                result = result_list[i]
                if result == 'Successful':
                    perturbed_all.append(perturbed_list[i])
                    label_all.append(int(ground_list[i]))
        return perturbed_all, label_all
