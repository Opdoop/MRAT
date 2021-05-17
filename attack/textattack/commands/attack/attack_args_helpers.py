"""

TextAttack Command Helpers for Attack
------------------------------------------

"""

import argparse
import copy
import importlib
import json
import os
import time

import textattack

from .attack_args import (
    ATTACK_RECIPE_NAMES,
    BLACK_BOX_TRANSFORMATION_CLASS_NAMES,
    CONSTRAINT_CLASS_NAMES,
    GOAL_FUNCTION_CLASS_NAMES,
    HUGGINGFACE_DATASET_BY_MODEL,
    SEARCH_METHOD_CLASS_NAMES,
    TEXTATTACK_DATASET_BY_MODEL,
    WHITE_BOX_TRANSFORMATION_CLASS_NAMES,
)

# The split token allows users to optionally pass multiple arguments in a single
# parameter by separating them with the split token.
ARGS_SPLIT_TOKEN = "^"


def add_model_args(parser):
    """Adds model-related arguments to an argparser.

    This is useful because we want to load pretrained models using
    multiple different parsers that share these, but not all, arguments.
    """
    model_group = parser.add_mutually_exclusive_group()

    model_names = list(HUGGINGFACE_DATASET_BY_MODEL.keys()) + list(
        TEXTATTACK_DATASET_BY_MODEL.keys()
    )
    model_group.add_argument(
        "--model",
        type=str,
        required=False,
        default=None,
        help='Name of or path to a pre-trained model to attack. Usage: "--model {model}:{arg_1}={value_1},{arg_3}={value_3},...". Choices: '
        + str(model_names),
    )
    model_group.add_argument(
        "--model-from-file",
        type=str,
        required=False,
        help="File of model and tokenizer to import.",
    )
    model_group.add_argument(
        "--model-from-huggingface",
        type=str,
        required=False,
        help="huggingface.co ID of pre-trained model to load",
    )


def add_dataset_args(parser):
    """Adds dataset-related arguments to an argparser.

    This is useful because we want to load pretrained models using
    multiple different parsers that share these, but not all, arguments.
    """
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--dataset-from-huggingface",
        type=str,
        required=False,
        default=None,
        help="Dataset to load from `datasets` repository.",
    )
    dataset_group.add_argument(
        "--dataset-from-file",
        type=str,
        required=False,
        default=None,
        help="Dataset to load from a file.",
    )
    parser.add_argument(
        "--shuffle",
        type=eval,
        required=False,
        choices=[True, False],
        default="True",
        help="Randomly shuffle the data before attacking",
    )
    parser.add_argument(
        "--num-examples",
        "-n",
        type=int,
        required=False,
        default="5",
        help="The number of examples to process, -1 for entire dataset",
    )

    parser.add_argument(
        "--num-examples-offset",
        "-o",
        type=int,
        required=False,
        default=0,
        help="The offset to start at in the dataset.",
    )


def load_module_from_file(file_path):
    """Uses ``importlib`` to dynamically open a file and load an object from
    it."""  # 这个 lib 好神奇啊，两行就实现了从文件中读取
    temp_module_name = f"temp_{time.time()}"
    colored_file_path = textattack.shared.utils.color_text(
        file_path, color="blue", method="ansi"
    )
    textattack.shared.logger.info(f"Loading module from `{colored_file_path}`.")
    # import pdb
    # pdb.set_trace()
    spec = importlib.util.spec_from_file_location(temp_module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_transformation_from_args(args, model_wrapper):
    transformation_name = args.transformation
    if ARGS_SPLIT_TOKEN in transformation_name:
        transformation_name, params = transformation_name.split(ARGS_SPLIT_TOKEN)

        if transformation_name in WHITE_BOX_TRANSFORMATION_CLASS_NAMES:
            transformation = eval(
                f"{WHITE_BOX_TRANSFORMATION_CLASS_NAMES[transformation_name]}(model_wrapper.model, {params})"
            )
        elif transformation_name in BLACK_BOX_TRANSFORMATION_CLASS_NAMES:
            transformation = eval(
                f"{BLACK_BOX_TRANSFORMATION_CLASS_NAMES[transformation_name]}({params})"
            )
        else:
            raise ValueError(f"Error: unsupported transformation {transformation_name}")
    else:
        if transformation_name in WHITE_BOX_TRANSFORMATION_CLASS_NAMES:
            transformation = eval(
                f"{WHITE_BOX_TRANSFORMATION_CLASS_NAMES[transformation_name]}(model_wrapper.model)"
            )
        elif transformation_name in BLACK_BOX_TRANSFORMATION_CLASS_NAMES:
            transformation = eval(
                f"{BLACK_BOX_TRANSFORMATION_CLASS_NAMES[transformation_name]}()"
            )
        else:
            raise ValueError(f"Error: unsupported transformation {transformation_name}")
    return transformation


def parse_goal_function_from_args(args, model):
    goal_function = args.goal_function
    if ARGS_SPLIT_TOKEN in goal_function:
        goal_function_name, params = goal_function.split(ARGS_SPLIT_TOKEN)
        if goal_function_name not in GOAL_FUNCTION_CLASS_NAMES:
            raise ValueError(f"Error: unsupported goal_function {goal_function_name}")
        goal_function = eval(
            f"{GOAL_FUNCTION_CLASS_NAMES[goal_function_name]}(model, {params})"
        )
    elif goal_function in GOAL_FUNCTION_CLASS_NAMES:
        goal_function = eval(f"{GOAL_FUNCTION_CLASS_NAMES[goal_function]}(model)")
    else:
        raise ValueError(f"Error: unsupported goal_function {goal_function}")
    goal_function.query_budget = args.query_budget
    goal_function.model_batch_size = args.model_batch_size
    goal_function.model_cache_size = args.model_cache_size
    return goal_function


def parse_constraints_from_args(args):
    if not args.constraints:
        return []

    _constraints = []
    for constraint in args.constraints:
        if ARGS_SPLIT_TOKEN in constraint:
            constraint_name, params = constraint.split(ARGS_SPLIT_TOKEN)
            if constraint_name not in CONSTRAINT_CLASS_NAMES:
                raise ValueError(f"Error: unsupported constraint {constraint_name}")
            _constraints.append(
                eval(f"{CONSTRAINT_CLASS_NAMES[constraint_name]}({params})")
            )
        elif constraint in CONSTRAINT_CLASS_NAMES:
            _constraints.append(eval(f"{CONSTRAINT_CLASS_NAMES[constraint]}()"))
        else:
            raise ValueError(f"Error: unsupported constraint {constraint}")

    return _constraints


def parse_attack_from_args(args):
    # 根据传入的参数，返回 attack 模型
    model = parse_model_from_args(args)  # 根据传入的参数，获取被攻击模型
    if args.recipe:  # 如果指定了 Attack recipe
        if ARGS_SPLIT_TOKEN in args.recipe:  # 如果 recipe 有 '^' 切断标记
            recipe_name, params = args.recipe.split(ARGS_SPLIT_TOKEN)  # 将 recipe 和 '^' 后的参数区 split ，获取到 recipe_name 和 params
            if recipe_name not in ATTACK_RECIPE_NAMES:  # 如果 recipe 不在预设范围内
                raise ValueError(f"Error: unsupported recipe {recipe_name}")  # 报错，说没找到
            recipe = eval(f"{ATTACK_RECIPE_NAMES[recipe_name]}.build(model, {params})")  # 使用 eval 创建 attack 模型，为何使用 eval 不清楚。
        elif args.recipe in ATTACK_RECIPE_NAMES:  # 单纯的传入了 recipe
            recipe = eval(f"{ATTACK_RECIPE_NAMES[args.recipe]}.build(model)")  # 使用 eval 创建 attack 模型
        else:  # 没用传入 recipe 参数
            raise ValueError(f"Invalid recipe {args.recipe}")  # 报错
        recipe.goal_function.query_budget = args.query_budget  # woc 找到了， 每个样本允许的最大 query 次数，默认是 inf
        recipe.goal_function.model_batch_size = args.model_batch_size  # 模型每个 batch 的大小
        recipe.goal_function.model_cache_size = args.model_cache_size  # The maximum number of items to keep in the model results cache at once， 这个  items 是啥啊？
        recipe.constraint_cache_size = args.constraint_cache_size  # The maximum number of items to keep in the constraints cache at once， 这 items 又是啥啊？
        print('Recipe:',args.recipe)
        return recipe  # 返回攻击模型
    elif args.attack_from_file:  # 如果从文件中读取 Attack，这段先不看了，跳过
        if ARGS_SPLIT_TOKEN in args.attack_from_file:
            attack_file, attack_name = args.attack_from_file.split(ARGS_SPLIT_TOKEN)
        else:
            attack_file, attack_name = args.attack_from_file, "attack"
        attack_module = load_module_from_file(attack_file)
        if not hasattr(attack_module, attack_name):
            raise ValueError(
                f"Loaded `{attack_file}` but could not find `{attack_name}`."
            )
        attack_func = getattr(attack_module, attack_name)
        return attack_func(model)
    else:  # 既不是读取内置的 recipe，也不是自定义文件，则从传入的参数中读取攻击的各部分
        goal_function = parse_goal_function_from_args(args, model)  # 读取目标函数
        transformation = parse_transformation_from_args(args, model)  # 读取变换方法
        constraints = parse_constraints_from_args(args)  # 读取约束
        if ARGS_SPLIT_TOKEN in args.search:  # 如果有 '^' 切割符
            search_name, params = args.search.split(ARGS_SPLIT_TOKEN)  # 在 '^' 处切分，得到搜索方法字符串和参数
            if search_name not in SEARCH_METHOD_CLASS_NAMES:  # 如果搜索方法不存在
                raise ValueError(f"Error: unsupported search {search_name}")  # 报错
            search_method = eval(f"{SEARCH_METHOD_CLASS_NAMES[search_name]}({params})")  # 正常，创建
        elif args.search in SEARCH_METHOD_CLASS_NAMES:  # 如果没有 '^' 切割符，且是预定义的搜索方法
            search_method = eval(f"{SEARCH_METHOD_CLASS_NAMES[args.search]}()")  # 直接尝试创建
        else:  # 没找到
            raise ValueError(f"Error: unsupported attack {args.search}")  # 报错
    return textattack.shared.Attack( #返回攻击模型，这里是从参数读取攻击策略各部分的 return ，调用 Attack 方法创建攻击模型
        goal_function,
        constraints,
        transformation,
        search_method,
        constraint_cache_size=args.constraint_cache_size,
    )


def parse_model_from_args(args):
    # 根据传入的参数，读取攻击的目标模型

    if args.model_from_file:  # 从指定的 .py 文件读取模型
        # Support loading the model from a .py file where a model wrapper
        # is instantiated.
        colored_model_name = textattack.shared.utils.color_text(  # 什么神奇的改变 std 字符串颜色的函数吧，没读过
            args.model_from_file, color="blue", method="ansi"
        )
        textattack.shared.logger.info(
            f"Loading model and tokenizer from file: {colored_model_name}"  # 打印传入的文件名
        )
        if ARGS_SPLIT_TOKEN in args.model_from_file:  # 如果有 '^' 分隔符，
            model_file, model_name = args.model_from_file.split(ARGS_SPLIT_TOKEN) #分别获取 model_file 和 model_name
        else: # 如果没传入 model_name
            _, model_name = args.model_from_file, "model"  # 给 model_name 赋个  "model" 字符串，这个 _ 参数有啥用呢？
        try:
            model_module = load_module_from_file(args.model_from_file)  # 尝试从文件内读取模型，这个 load 函数有空再读读
        except Exception:
            raise ValueError(f"Failed to import file {args.model_from_file}")  # 如果读取失败，报错
        try:
            model = getattr(model_module, model_name)  # 尝试从创建的 attack 模型对象中获取其 model_name 参数
        except AttributeError:
            raise AttributeError(
                f"``{model_name}`` not found in module {args.model_from_file}"  # 从文件中读取的模型没有 model_name 参数
            )

        # if not isinstance(model, textattack.models.wrappers.ModelWrapper):  # 如果读到的不是个 modelwrapper 的对象
        #     raise TypeError(
        #         "Model must be of type "
        #         f"``textattack.models.ModelWrapper``, got type {type(model)}"  # 报 类型 错
        #     )
    elif (args.model in HUGGINGFACE_DATASET_BY_MODEL) or args.model_from_huggingface:  # 如果 model 是 textattack 提供的，或者是从 huggingface 载入其他的预训练模型
        # Support loading models automatically from the HuggingFace model hub.
        import transformers  # 导入 transformer

        model_name = (
            HUGGINGFACE_DATASET_BY_MODEL[args.model][0]
            if (args.model in HUGGINGFACE_DATASET_BY_MODEL)
            else args.model_from_huggingface
        )  # 获取 model name

        if ARGS_SPLIT_TOKEN in model_name:
            model_class, model_name = model_name
            model_class = eval(f"transformers.{model_class}")  # model class 干啥用的？ 不太懂
        else:
            model_class, model_name = (
                transformers.AutoModelForSequenceClassification,
                model_name,
            )
        colored_model_name = textattack.shared.utils.color_text(
            model_name, color="blue", method="ansi"
        )
        textattack.shared.logger.info(
            f"Loading pre-trained model from HuggingFace model repository: {colored_model_name}"
        )
        model = model_class.from_pretrained(model_name)  # 载入预训练模型
        tokenizer = textattack.models.tokenizers.AutoTokenizer(model_name)  # 载入分词器
        model = textattack.models.wrappers.HuggingFaceModelWrapper(  # 创建为 ModelWrapper 类
            model, tokenizer, batch_size=args.model_batch_size
        )
    elif args.model in TEXTATTACK_DATASET_BY_MODEL:  # 如果 model 是预训练的其他模型
        # Support loading TextAttack pre-trained models via just a keyword.
        model_path, _ = TEXTATTACK_DATASET_BY_MODEL[args.model]
        model = textattack.shared.utils.load_textattack_model_from_path(
            args.model, model_path
        )
        # Choose the approprate model wrapper (based on whether or not this is
        # a HuggingFace model).
        if isinstance(model, textattack.models.helpers.T5ForTextToText):  # 如果是 T5 模型
            model = textattack.models.wrappers.HuggingFaceModelWrapper(
                model, model.tokenizer, batch_size=args.model_batch_size
            )
        else:  # CNN 或 LSTM 模型
            model = textattack.models.wrappers.PyTorchModelWrapper(
                model, model.tokenizer, batch_size=args.model_batch_size
            )
    elif args.model and os.path.exists(args.model):
        # 通过 TextAttack 训练的模型，只需要提供路径
        # Support loading TextAttack-trained models via just their folder path.
        # If `args.model` is a path/directory, let's assume it was a model
        # trained with textattack, and try and load it.
        model_args_json_path = os.path.join(args.model, "train_args.json")  # 如果有这个 json 文件
        if not os.path.exists(model_args_json_path):
            raise FileNotFoundError(
                f"Tried to load model from path {args.model} - could not find train_args.json."
            )
        model_train_args = json.loads(open(model_args_json_path).read())  # 读取参数
        if model_train_args["model"] not in {"cnn", "lstm"}:
            # for huggingface models, set args.model to the path of the model
            model_train_args["model"] = args.model  # 如果不是 cnn/lstm 模型，赋值为传入的模型文件夹路径
        num_labels = model_train_args["num_labels"]  # 获取分类任务的类别数量
        from textattack.commands.train_model.train_args_helpers import model_from_args
        # 载入模型
        model = model_from_args(
            argparse.Namespace(**model_train_args),
            num_labels,
            model_path=args.model,
        )

    else:
        raise ValueError(f"Error: unsupported TextAttack model {args.model}")

    return model


def parse_dataset_from_args(args):
    # Automatically detect dataset for huggingface & textattack models.
    # This allows us to use the --model shortcut without specifying a dataset.
    # 根据传入的参数读取数据集
    if args.model in HUGGINGFACE_DATASET_BY_MODEL:  # 如果是提供的预训练模型，则读取预训练时候使用的数据集
        pass # 使用传入的 dataset_from_huggingface
        # _, args.dataset_from_huggingface = HUGGINGFACE_DATASET_BY_MODEL[args.model]  # 读取 huggingface 的数据集
    elif args.model in TEXTATTACK_DATASET_BY_MODEL:  # 如果是预训练的 cnn/lstm/t5
        _, dataset = TEXTATTACK_DATASET_BY_MODEL[args.model]  # 也读取预训练时候使用的数据集
        if dataset[0].startswith("textattack"):  # 不推荐的读取方法
            # unsavory way to pass custom dataset classes
            # ex: dataset = ('textattack.datasets.translation.TedMultiTranslationDataset', 'en', 'de')
            dataset = eval(f"{dataset[0]}")(*dataset[1:])
            return dataset
        else:
            args.dataset_from_huggingface = dataset  # 返回数据集
    # Automatically detect dataset for models trained with textattack.
    # 使用 textattack 训练的模型，可用从 train_args.json 中自动读取训练时使用的数据集
    elif args.model and os.path.exists(args.model):  # 传入了模型路径，并且路径存在
        model_args_json_path = os.path.join(args.model, "train_args.json")  # 从保存的文件中读取模型参数
        if not os.path.exists(model_args_json_path):  # 如果参数文件不存在
            raise FileNotFoundError(
                f"Tried to load model from path {args.model} - could not find train_args.json."  # 报错
            )
        model_train_args = json.loads(open(model_args_json_path).read())  # 读取文件
        try:  # 又是 split 那一套
            if ARGS_SPLIT_TOKEN in model_train_args["dataset"]:
                name, subset = model_train_args["dataset"].split(ARGS_SPLIT_TOKEN)
            else:
                name, subset = model_train_args["dataset"], None
            # 不使用 train_args.json 中的 dev_split，使用手工指定的 arg.dataset_from_huggingface
            # args.dataset_from_huggingface = (  # 载入模型
            #     name,
            #     subset,
            #     model_train_args["dataset_dev_split"],
            # )
        except KeyError:
            raise KeyError(
                f"Tried to load model from path {args.model} but can't initialize dataset from train_args.json."  # 载入失败，报错
            )

    # Get dataset from args.
    # 从自定义的 .py 文件中载入模型
    if args.dataset_from_file:
        textattack.shared.logger.info(
            f"Loading model and tokenizer from file: {args.model}"
        )
        if ARGS_SPLIT_TOKEN in args.dataset_from_file:
            dataset_file, dataset_name = args.dataset_from_file.split(ARGS_SPLIT_TOKEN)
        else:
            dataset_file, dataset_name = args.dataset_from_file, "dataset"
        try:
            dataset_module = load_module_from_file(dataset_file)   # 从文件中载入数据
        except Exception:
            raise ValueError(
                f"Failed to import dataset from file {args.dataset_from_file}"  # 载入失败
            )
        try:
            dataset = getattr(dataset_module, dataset_name)  # 获取 dataset_name 赋值给 dataset
        except AttributeError:
            raise AttributeError(
                f"``dataset`` not found in module {args.dataset_from_file}"  # 没读到，报错
            )
    # 从 huggingface dataset 读取数据集
    elif args.dataset_from_huggingface:
        dataset_args = args.dataset_from_huggingface
        if isinstance(dataset_args, str):
            if ARGS_SPLIT_TOKEN in dataset_args:
                dataset_args = dataset_args.split(ARGS_SPLIT_TOKEN)
                if 'None' in dataset_args: # 修正 huggingface datasets 中 loaddataset 的 subset 传入 'None' 报错，改为 None
                    dataset_args[1] = None
            else:
                dataset_args = (dataset_args,)
        dataset = textattack.datasets.HuggingFaceDataset(
            *dataset_args, shuffle=args.shuffle
        )
        dataset.examples = dataset.examples[args.num_examples_offset :]
    else:
        raise ValueError("Must supply pretrained model or dataset")
    return dataset


def parse_logger_from_args(args):
    # 根据参数，创建 logger
    # Create logger
    attack_log_manager = textattack.loggers.AttackLogManager()  # 创建一个标准 logger

    # 下面一堆都是给 logger 设置输出文件地址和类型的（txt/csv），一般不会改。
    # Get current time for file naming
    timestamp = time.strftime("%Y-%m-%d-%H-%M")  # 获取时间戳

    # Get default directory to save results
    current_dir = os.path.dirname(os.path.realpath(__file__))  # 获取当前文件所属路径
    outputs_dir = os.path.join(
        current_dir, os.pardir, os.pardir, os.pardir, os.pardir, "outputs", "attacks"  # 创建 logger 保存结果的路径
    )
    out_dir_txt = out_dir_csv = os.path.normpath(outputs_dir)  # 将拼接的输出路径标准化

    # Get default txt and csv file names
    # 设置默认的文件名
    if args.recipe:  # 如果有 recipe 参数
        filename_txt = f"{args.model}_{args.recipe}_{timestamp}.txt"
        filename_csv = f"{args.model}_{args.recipe}_{timestamp}.csv"
    else:
        filename_txt = f"{args.model}_{timestamp}.txt"
        filename_csv = f"{args.model}_{timestamp}.csv"

    # if '--log-to-txt' specified with arguments
    #  如果设置了 logger 结果保存为 txt 文件
    if args.log_to_txt:
        # if user decide to save to a specific directory
        # 如果在参数中指定了具体路径
        if args.log_to_txt[-1] == "/":
            out_dir_txt = args.log_to_txt
        # else if path + filename is given
        # 如果是路径+文件名
        elif args.log_to_txt[-4:] == ".txt":
            out_dir_txt = args.log_to_txt.rsplit("/", 1)[0]
            filename_txt = args.log_to_txt.rsplit("/", 1)[-1]
        # otherwise, customize filename
        # 仅有文件名
        else:
            filename_txt = f"{args.log_to_txt}.txt"

    # if "--log-to-csv" is called
    # 如果设置了保存到 csv 文件
    if args.log_to_csv:
        # if user decide to save to a specific directory
        # 仅有路径
        if args.log_to_csv[-1] == "/":
            out_dir_csv = args.log_to_csv
        # else if path + filename is given
        # 路径 + 文件名
        elif args.log_to_csv[-4:] == ".csv":
            # out_dir_csv = args.log_to_csv.rsplit("/", 1)[0]
            # filename_csv = args.log_to_csv.rsplit("/", 1)[-1]
            # 仅传入 csv 文件名，使用标准输出 out_dir
            filename_csv = args.log_to_csv
        # otherwise, customize filename
        # 仅有文件名
        else:
            filename_csv = f"{args.log_to_csv}.csv"

    # in case directory doesn't exist
    # 如果保存的路径不存在，创建路径
    if not os.path.exists(out_dir_txt):
        os.makedirs(out_dir_txt)
    if not os.path.exists(out_dir_csv):
        os.makedirs(out_dir_csv)

    # if "--log-to-txt" specified in terminal command (with or without arg), save to a txt file
    # 如果仅给了参数  log-to-txt，没给值
    if args.log_to_txt == "" or args.log_to_txt:
        attack_log_manager.add_output_file(os.path.join(out_dir_txt, filename_txt))  # 文件保存路径添加给 logger

    # if "--log-to-csv" specified in terminal command(with  or without arg), save to a csv file
    if args.log_to_csv == "" or args.log_to_csv:
        # "--csv-style used to swtich from 'fancy' to 'plain'
        color_method = None if args.csv_style == "plain" else "file"  # 设置 csv 的 color method
        csv_path = os.path.join(out_dir_csv, filename_csv)  # 设置 csv 文件的保存路径
        attack_log_manager.add_output_csv(csv_path, color_method)  # 将保存路径和 color method 添加给 logger
        textattack.shared.logger.info(f"Logging to CSV at path {csv_path}.")  # 打印日志

    # Visdom
    if args.enable_visdom:
        attack_log_manager.enable_visdom()  # 不知道 visdom 是啥东西

    # Weights & Biases
    if args.enable_wandb:
        attack_log_manager.enable_wandb()  # 这个也不清楚

    # Stdout
    if not args.disable_stdout:  # 如果要在屏幕输出
        attack_log_manager.enable_stdout()  # 开启屏幕输出
    return attack_log_manager


def parse_checkpoint_from_args(args):
    file_name = os.path.basename(args.checkpoint_file)
    if file_name.lower() == "latest":
        dir_path = os.path.dirname(args.checkpoint_file)
        chkpt_file_names = [f for f in os.listdir(dir_path) if f.endswith(".ta.chkpt")]
        assert chkpt_file_names, "Checkpoint directory is empty"
        timestamps = [int(f.replace(".ta.chkpt", "")) for f in chkpt_file_names]
        latest_file = str(max(timestamps)) + ".ta.chkpt"
        checkpoint_path = os.path.join(dir_path, latest_file)
    else:
        checkpoint_path = args.checkpoint_file

    checkpoint = textattack.shared.Checkpoint.load(checkpoint_path)

    return checkpoint


def default_checkpoint_dir():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    checkpoints_dir = os.path.join(
        current_dir, os.pardir, os.pardir, os.pardir, "checkpoints"
    )
    return os.path.normpath(checkpoints_dir)


def merge_checkpoint_args(saved_args, cmdline_args):
    """Merge previously saved arguments for checkpoint and newly entered
    arguments."""
    args = copy.deepcopy(saved_args)
    # Newly entered arguments take precedence
    args.parallel = cmdline_args.parallel
    # If set, replace
    if cmdline_args.checkpoint_dir:
        args.checkpoint_dir = cmdline_args.checkpoint_dir
    if cmdline_args.checkpoint_interval:
        args.checkpoint_interval = cmdline_args.checkpoint_interval

    return args
