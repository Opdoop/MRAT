import os

import train.shared
from train.datasets_wrapper import HuggingFaceDataset
import re
from train import models


logger = train.shared.logger
ARGS_SPLIT_TOKEN = "^"

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def prepare_dataset_for_training(datasets_dataset):
    """
    Changes an `datasets` dataset into the proper format for
    tokenization."""

    def prepare_example_dict(ex):
        """Returns the values in order corresponding to the data.

        ex:
            'Some text input'
        or in the case of multi-sequence inputs:
            ('The premise', 'the hypothesis',)
        etc.
        """
        values = list(ex.values())
        if len(values) == 1:
            return values[0]
        return tuple(values)

    text, outputs = zip(*((prepare_example_dict(x[0]), x[1]) for x in datasets_dataset))
    # not use preprocess, consist with attack scenario. But lead model normal acc decrease 1-2 point.
    # text = [clean_str(t) for t in text]
    return list(text), list(outputs)


def dataset_from_args(args):
    """Returns a tuple of ``HuggingFaceDataset`` for the train and test
    datasets for ``args.dataset``.
    """
    dataset_args = args.dataset.split(ARGS_SPLIT_TOKEN)

    if args.dataset_train_split:
        train_dataset = HuggingFaceDataset(
            *dataset_args, split=args.dataset_train_split
        )
    else:
        try:
            train_dataset = HuggingFaceDataset(
                *dataset_args, split="train"
            )
            args.dataset_train_split = "train"
        except KeyError:
            raise KeyError(f"Error: no `train` split found in `{args.dataset}` dataset")
    train_text, train_labels = prepare_dataset_for_training(train_dataset)

    if args.dataset_dev_split:
        eval_dataset = HuggingFaceDataset(
            *dataset_args, split=args.dataset_dev_split
        )
    else:
        # try common dev split names
        try:
            eval_dataset = HuggingFaceDataset(
                *dataset_args, split="dev"
            )
            args.dataset_dev_split = "dev"
        except KeyError:
            try:
                eval_dataset = HuggingFaceDataset(
                    *dataset_args, split="eval"
                )
                args.dataset_dev_split = "eval"
            except KeyError:
                try:
                    eval_dataset = HuggingFaceDataset(
                        *dataset_args, split="validation"
                    )
                    args.dataset_dev_split = "validation"
                except KeyError:
                    try:
                        eval_dataset = HuggingFaceDataset(
                            *dataset_args, split="test"
                        )
                        args.dataset_dev_split = "test"
                    except KeyError:
                        raise KeyError(
                            f"Could not find `dev`, `eval`, `validation`, or `test` split in dataset {args.dataset}."
                        )
    eval_text, eval_labels = prepare_dataset_for_training(eval_dataset)

    return train_text, train_labels, eval_text, eval_labels

def dataset_from_local(args):
    train_text, train_labels, eval_text, eval_labels = [], [], [], []
    if args.dataset in ['mr' or 'imdb']: # single sentence/document input
        with open(args.train_path, 'r', encoding='utf8') as fin:
            for line in fin.readlines():
                label, string = int(line[0]), line[2:].strip()
                train_text.append(string)
                train_labels.append(label)

        with open(args.eval_path, 'r', encoding='utf8') as fin:
            for line in fin.readlines():
                label, string = int(line[0]), line[2:].strip()
                eval_text.append(string)
                eval_labels.append(label)
    else:
        def read_data(filepath):
            import collections
            """
            Read the premises, hypotheses and labels from some NLI dataset's
            file and return them in a dictionary. The file should be in the same
            form as SNLI's .txt files.

            Args:
                filepath: The path to a file containing some premises, hypotheses
                    and labels that must be read. The file should be formatted in
                    the same way as the SNLI (and MultiNLI) dataset.

            Returns:
                A dictionary containing three lists, one for the premises, one for
                the hypotheses, and one for the labels in the input data.
            """
            label2id = {'entailment': 1,
                        'neutral': 2,
                        'contradiction': 0}  # consist with  hugging face datasets
            import re
            _RE_COMBINE_WHITESPACE = re.compile(r"\s+")

            with open(filepath, 'r', encoding='utf8') as input_data:
                inputs, labels = [], []

                # Translation tables to remove parentheses and punctuation from
                # strings.
                parentheses_table = str.maketrans({'(': None, ')': None})

                # Ignore the headers on the first line of the file.
                next(input_data)

                for line in input_data:
                    line = line.strip().split('\t')

                    # Ignore sentences that have no gold label.
                    if line[0] == '-':
                        continue

                    premise = line[1]
                    hypothesis = line[2]

                    # Remove '(' and ')' from the premises and hypotheses.
                    premise = premise.translate(parentheses_table)
                    hypothesis = hypothesis.translate(parentheses_table)

                    # Substitute multiple space to one
                    premise = _RE_COMBINE_WHITESPACE.sub(" ", premise).strip()
                    hypothesis = _RE_COMBINE_WHITESPACE.sub(" ", hypothesis).strip()

                    # input = collections.OrderedDict(
                    #     [('premise', premise),
                    #      ('hypothesis', hypothesis)]
                    # )
                    input = (premise, hypothesis)
                    inputs.append(input)

                    labels.append(label2id[line[0]])

                return inputs, labels

        train_text, train_labels = read_data(args.train_path)
        eval_text, eval_labels = read_data(args.eval_path)

    return train_text, train_labels, eval_text, eval_labels




def model_from_args(train_args, num_labels, model_path=None):
    """Constructs a model from its `train_args.json`.
    If huggingface model, loads from model hub address. If TextAttack
    lstm/cnn, loads from disk (and `model_path` provides the path to the
    model).
    """
    if train_args.model == "lstm":
        train.shared.logger.info("Loading model: LSTMForClassification")
        model = models.helpers.LSTMForClassification(
            max_seq_length=train_args.max_length,
            num_labels=num_labels,
            emb_layer_trainable=False,
        )
        if model_path:
            model.load_from_disk(model_path)

        model = models.wrappers.PyTorchModelWrapper(model, model.tokenizer)
    elif train_args.model == "cnn":
        train.shared.logger.info(
            "Loading model: WordCNNForClassification"
        )
        model = models.helpers.WordCNNForClassification(
            max_seq_length=train_args.max_length,
            num_labels=num_labels,
            emb_layer_trainable=False,
        )
        if model_path:
            model.load_from_disk(model_path)

        model = models.wrappers.PyTorchModelWrapper(model, model.tokenizer)
    else:
        import transformers

        train.shared.logger.info(
            f"Loading transformers AutoModelForSequenceClassification: {train_args.model}"
        )

        if train_args.mixup_training or train_args.adversarial_training:
            model = models.helpers.MixBert(train_args.model, num_labels=num_labels, finetuning_task=train_args.dataset )
        else:
            config = transformers.AutoConfig.from_pretrained(
                train_args.model, num_labels=num_labels, finetuning_task=train_args.dataset
            )
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                train_args.model,
                config=config,
            )
        tokenizer = models.tokenizers.AutoTokenizer(
            train_args.model, use_fast=True, max_length=train_args.max_length
        )

        if model_path:
            model.load_from_disk(model_path)

        model = models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
    return model


def write_readme(args, best_eval_score, best_eval_score_epoch):
    # Save args to file
    readme_save_path = os.path.join(args.output_dir, "README.md")
    dataset_name = (
        args.dataset.split(ARGS_SPLIT_TOKEN)[0]
        if ARGS_SPLIT_TOKEN in args.dataset
        else args.dataset
    )
    task_name = "classification"
    loss_func = "mean squared error"
    metric_name = "accuracy"
    epoch_info = f"{best_eval_score_epoch} epoch" + (
        "s" if best_eval_score_epoch > 1 else ""
    )

    readme_text = f"""
## Model Card

This `{args.model}` model was trained or fine-tuned for sequence classification 
and the {dataset_name} dataset loaded using the `datasets` library. The model was trained of fine-tuned
for {args.num_train_epochs} epochs with a batch size of {args.batch_size}, a learning
rate of {args.learning_rate}, and a maximum sequence length of {args.max_length}.
Since this was a {task_name} task, the model was trained with a {loss_func} loss function.
The best score the model achieved on this task was {best_eval_score}, as measured by the
eval set {metric_name}, found after {epoch_info}.

"""

    with open(readme_save_path, "w", encoding="utf-8") as f:
        f.write(readme_text.strip() + "\n")
    logger.info(f"Wrote README to {readme_save_path}.")
