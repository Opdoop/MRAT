import os

import textattack
from textattack.commands.attack.attack_args import ATTACK_RECIPE_NAMES
from textattack.commands.attack.attack_args_helpers import ARGS_SPLIT_TOKEN
from textattack.commands.augment import AUGMENTATION_RECIPE_NAMES
import re

logger = textattack.shared.logger


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
    将 huggingface 数据集转换格式，转换为合适 tokenizer 处理的格式
    # 这应该是啥也没做，就是把单输入和多输入的数据集进行了标准化
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

    # 这应该是啥也没做，就是把单输入和多输入的数据集进行了标准化
    text, outputs = zip(*((prepare_example_dict(x[0]), x[1]) for x in datasets_dataset))
    # import pdb
    # pdb.set_trace()
    # text = [clean_str(t) for t in text]  # 与攻击时统一。虽然会导致被攻击模型 acc 下降 1-2 个点。
    return list(text), list(outputs)


def dataset_from_args(args):
    """Returns a tuple of ``HuggingFaceDataset`` for the train and test
    datasets for ``args.dataset``.
    根据 'args.dataset' 返回 'HuggingFaceDataset' 的训练集和测试集的 tuple
    """
    dataset_args = args.dataset.split(ARGS_SPLIT_TOKEN)

    # 获取训练集
    if args.dataset_train_split:
        train_dataset = textattack.datasets.HuggingFaceDataset(
            *dataset_args, split=args.dataset_train_split
        )
    else:
        try:
            train_dataset = textattack.datasets.HuggingFaceDataset(
                *dataset_args, split="train"
            )
            args.dataset_train_split = "train"
        except KeyError:
            raise KeyError(f"Error: no `train` split found in `{args.dataset}` dataset")
    train_text, train_labels = prepare_dataset_for_training(train_dataset) #传入 HuggingFaceDataset，返回 train_text 和 train_labels 的 list

    # 获取测试集，测试集就分别尝试 dev/eval/validation/test 有哪个就读哪个。有覆盖，最后这四种 split ，读到的作为测试集。
    if args.dataset_dev_split:
        eval_dataset = textattack.datasets.HuggingFaceDataset(
            *dataset_args, split=args.dataset_dev_split
        )
    else:
        # try common dev split names
        try:
            eval_dataset = textattack.datasets.HuggingFaceDataset(
                *dataset_args, split="dev"
            )
            args.dataset_dev_split = "dev"
        except KeyError:
            try:
                eval_dataset = textattack.datasets.HuggingFaceDataset(
                    *dataset_args, split="eval"
                )
                args.dataset_dev_split = "eval"
            except KeyError:
                try:
                    eval_dataset = textattack.datasets.HuggingFaceDataset(
                        *dataset_args, split="validation"
                    )
                    args.dataset_dev_split = "validation"
                except KeyError:
                    try:
                        eval_dataset = textattack.datasets.HuggingFaceDataset(
                            *dataset_args, split="test"
                        )
                        args.dataset_dev_split = "test"
                    except KeyError:
                        raise KeyError(
                            f"Could not find `dev`, `eval`, `validation`, or `test` split in dataset {args.dataset}."
                        )
    eval_text, eval_labels = prepare_dataset_for_training(eval_dataset) # 标准化

    return train_text, train_labels, eval_text, eval_labels


def model_from_args(train_args, num_labels, model_path=None):
    """Constructs a model from its `train_args.json`.
    根据参数创建模型。（只有 run_training 用到了这个函数，那 attack 的时候载入模型用啥函数啊？）（都用了，attack 也这样读取 victim 模型
    若是 TextAttack 的 lstm/cnn 模型，则通过 textattack.models 进行创建
    若是 huggingface 的 bert-based 模型，则传入模型名称，从 huggingface model hub 上载入
    如果设置了 model_path ，则从本地载入权重

    If huggingface model, loads from model hub address. If TextAttack
    lstm/cnn, loads from disk (and `model_path` provides the path to the
    model).
    """
    if train_args.model == "lstm":
        # 传入模型类型为 lstm
        textattack.shared.logger.info("Loading textattack model: LSTMForClassification")
        model = textattack.models.helpers.LSTMForClassification(
            max_seq_length=train_args.max_length, # 原来 max_length 是在模型里设置，而不是在 dataloader 里设置的
            num_labels=num_labels, # 类别数量
            emb_layer_trainable=False, # embedding 不微调
        )
        if model_path:
            model.load_from_disk(model_path)

        model = textattack.models.wrappers.PyTorchModelWrapper(model, model.tokenizer) # 这 tokenizer 是继承自 nn.Model 的
    elif train_args.model == "cnn":
        # cnn 同理
        textattack.shared.logger.info(
            "Loading textattack model: WordCNNForClassification"
        )
        model = textattack.models.helpers.WordCNNForClassification(
            max_seq_length=train_args.max_length,
            num_labels=num_labels,
            emb_layer_trainable=False,
        )
        if model_path:
            model.load_from_disk(model_path)

        model = textattack.models.wrappers.PyTorchModelWrapper(model, model.tokenizer)  # 把 tokenizer 和 model 打包在一起
    else:
        # 如果是 huggingface 的模型
        import transformers

        textattack.shared.logger.info(
            f"Loading transformers AutoModelForSequenceClassification: {train_args.model}"
        )
        # 创建 AutoConfig
        config = transformers.AutoConfig.from_pretrained(
            train_args.model, num_labels=num_labels, finetuning_task=train_args.dataset  # finetune 过的数据集
        )
        model = transformers.AutoModelForSequenceClassification.from_pretrained(  # 序列分类模型
            train_args.model,
            config=config,
        )
        tokenizer = textattack.models.tokenizers.AutoTokenizer(
            train_args.model, use_fast=True, max_length=train_args.max_length  # 这里 max_length 是给 tokenizer 的，
        )

        model = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)  # 把 tokenizer 和 model 打包在一起

    return model


def attack_from_args(args):
    # 从参数中生成对应的攻击策略
    # 由于真的攻击对象需要给定模型，即在模型初始化之后才能生成对象。这里返回的是攻击的类型
    # note that this returns a recipe type, not an object
    # (we need to wait to have access to the model to initialize)
    attack_class = None
    if args.attack: # 检车是否是内置的攻击策略
        if args.attack in ATTACK_RECIPE_NAMES:
            attack_class = eval(ATTACK_RECIPE_NAMES[args.attack]) # eval 函数是啥用处啊
        else:
            raise ValueError(f"Unrecognized attack recipe: {args.attack}")

    # check attack-related args
    # 对抗训练要求至少进行几次在正常样本的预训练
    assert args.num_clean_epochs > 0, "--num-clean-epochs must be > 0"
    # 若要验证 robustness，必须指定 attack 类型
    assert not (
        args.check_robustness and not (args.attack)
    ), "--check_robustness must be used with --attack"

    return attack_class


def augmenter_from_args(args):
    # 从参数中生成对应的数据增强策略
    augmenter = None
    if args.augment:
        if args.augment in AUGMENTATION_RECIPE_NAMES:
            augmenter = eval(AUGMENTATION_RECIPE_NAMES[args.augment])(
                pct_words_to_swap=args.pct_words_to_swap,
                transformations_per_example=args.transformations_per_example,
            )
        else:
            raise ValueError(f"Unrecognized augmentation recipe: {args.augment}")
    return augmenter


def write_readme(args, best_eval_score, best_eval_score_epoch):
    # Save args to file
    # 保存参数到文件
    readme_save_path = os.path.join(args.output_dir, "README.md") # 文件名
    dataset_name = (
        args.dataset.split(ARGS_SPLIT_TOKEN)[0]
        if ARGS_SPLIT_TOKEN in args.dataset
        else args.dataset
    )  # 获取 split 或 dataset，这是不是有问题，难道不应该获取 split 和 dataset 吗
    task_name = "regression" if args.do_regression else "classification" # 任务类型，标签为整数就是分类，为浮点数就是回归
    loss_func = "mean squared error" if args.do_regression else "cross-entropy"  # 损失函数，分类是交叉熵，回归是最小二乘
    metric_name = "pearson correlation" if args.do_regression else "accuracy" # 评价指标，分类是准确率，回归是皮尔森系数
    epoch_info = f"{best_eval_score_epoch} epoch" + ( # 最优模型的轮数
        "s" if best_eval_score_epoch > 1 else ""
    )
    # 文本内容
    readme_text = f"""
## TextAttack Model Card

This `{args.model}` model was fine-tuned for sequence classification using TextAttack
and the {dataset_name} dataset loaded using the `datasets` library. The model was fine-tuned
for {args.num_train_epochs} epochs with a batch size of {args.batch_size}, a learning
rate of {args.learning_rate}, and a maximum sequence length of {args.max_length}.
Since this was a {task_name} task, the model was trained with a {loss_func} loss function.
The best score the model achieved on this task was {best_eval_score}, as measured by the
eval set {metric_name}, found after {epoch_info}.

For more information, check out [TextAttack on Github](https://github.com/QData/TextAttack).

"""
# 这怎么上来就 finetune 啊，貌似是给 bert_based 模型用的
    with open(readme_save_path, "w", encoding="utf-8") as f:
        f.write(readme_text.strip() + "\n")
    logger.info(f"Wrote README to {readme_save_path}.")
