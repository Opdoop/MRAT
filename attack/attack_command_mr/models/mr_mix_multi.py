import os
import json
import argparse
import sys

def _path():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    outputs_dir = os.path.realpath(os.path.join(
        current_dir, os.pardir, os.pardir, os.pardir
    ))
    return outputs_dir

def load_model_wrapper(model_name):
    # Support loading TextAttack-trained models via just their folder path.
    # If `args.model` is a path/directory, let's assume it was a model
    # trained with textattack, and try and load it.
    current_dir = os.path.dirname(os.path.realpath(__file__))
    model = os.path.realpath(os.path.join(
        current_dir, os.pardir, os.pardir, os.pardir, 'outputs', 'training', model_name
    ))
    model_args_json_path = os.path.join(model, "train_args.json")
    if not os.path.exists(model_args_json_path):
        raise FileNotFoundError(
            f"Tried to load model from path {model} - could not find train_args.json."
        )
    model_train_args = json.loads(open(model_args_json_path).read())
    if model_train_args["model"] not in {"cnn", "lstm"}:
        # for huggingface models, set args.model to the path of the model
        model_train_args["model"] = model
    num_labels = model_train_args["num_labels"]

    sys.path.append(_path())
    from train.train_model.train_args_helpers import model_from_args
    model = model_from_args(
        argparse.Namespace(**model_train_args),
        num_labels,
        model_path=model,
    )

    return model


model_name = f'bert-base-uncased-mr-mix-multi'
model_wrapper= load_model_wrapper(model_name)

tokenizer = model_wrapper.tokenizer
model = model_wrapper