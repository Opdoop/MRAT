import sys
import argparse
import os

def arg_wrapper():
    parser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="directory of model to train",
    )
    parser.add_argument(
        "--dataset-from-huggingface",
        type=str,
        default="",
        help="Dataset to load from `datasets` repository.",
    )
    parser.add_argument(
        "--recipe",
        "--attack-recipe",
        "-r",
        type=str,
        required=False,
        default="",
        help="full attack recipe (overrides provided goal function, transformation & constraints)",
    )
    parser.add_argument(
        "--query-budget",
        "-q",
        type=int,
        default=float("inf"),
        help="The maximum number of model queries allowed per example attacked.",
    )
    parser.add_argument(
        "--shuffle",
        type=eval,
        required=False,
        choices=[True, False],
        default="False",
        help="Randomly shuffle the data before attacking",
    )
    parser.add_argument(
        "--num-examples",
        "-n",
        type=int,
        required=False,
        default=-1,
        help="The number of examples to process, -1 for entire dataset",
    )

    parser.add_argument(
        "--log-to-csv",
        nargs="?",
        default=None,
        const="",
        type=str,
        help="Save attack logs to <install-dir>/outputs/~ by default; Include '/' at the end of argument to save "
             "output to specified directory in default naming convention; otherwise enter argument to specify "
             "file name",
    )

    parser.add_argument(
        "--csv-style",
        default='plain',
        const="fancy",
        nargs="?",
        type=str,
        help="Use --csv-style plain to remove [[]] around words",
    )

    parser.add_argument(
        "--disable-stdout", action="store_true", default=True, help="Disable logging to stdout"
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Run attack using multiple GPUs.",
    )

    parser.add_argument("--random-seed", default=21, type=int)


    parser.add_argument(
        "--checkpoint-interval",
        required=False,
        type=int,
        help="If set, checkpoint will be saved after attacking every N examples. If not set, no checkpoints will be saved.",
    )

    parser.add_argument(
        "--model-batch-size",
        type=int,
        default=32,
        help="The batch size for making calls to the model.",
    )
    parser.add_argument(
        "--model-cache-size",
        type=int,
        default=2 ** 18,
        help="The maximum number of items to keep in the model results cache at once.",
    )
    parser.add_argument(
        "--constraint-cache-size",
        type=int,
        default=2 ** 18,
        help="The maximum number of items to keep in the constraints cache at once.",
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="Whether to run attacks interactively.",
    )
    parser.add_argument(
        "--dataset-from-file",
        type=str,
        required=False,
        default=None,
        help="dataset to load from file",
    )
    parser.add_argument(
        "--num-examples-offset",
        "-o",
        type=int,
        required=False,
        default=0,
        help="The offset to start at in the dataset.",
    )
    parser.add_argument(
        "--model-from-file",
        type=str,
        required=False,
        help="File of model and tokenizer to import.",
    )
    parser.add_argument(
        "--model-from-huggingface",
        type=str,
        required=False,
        help="huggingface.co ID of pre-trained model to load",
    )
    parser.add_argument(
        "--log-to-txt",
        "-l",
        nargs="?",
        default=None,
        const="",
        type=str,
        help="Save attack logs to <install-dir>/outputs/~ by default; Include '/' at the end of argument to save "
             "output to specified directory in default naming convention; otherwise enter argument to specify "
             "file name",
    )
    parser.add_argument(
        "--enable-visdom", action="store_true", help="Enable logging to visdom."
    )
    parser.add_argument(
        "--enable-wandb",
        action="store_true",
        help="Enable logging to Weights & Biases.",
    )
    parser.add_argument(
        "--attack-n",
        action="store_true",
        default=False,
        help="Whether to run attack until `n` examples have been attacked (not skipped).",
    )
    args = parser.parse_args()
    return args

def _path():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    outputs_dir = os.path.join(
        current_dir, os.pardir
    )
    return outputs_dir

def model_path(model_name):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    path = os.path.realpath(os.path.join(
        current_dir, os.pardir, os.pardir, 'outputs', 'training', model_name
    ))
    return path

