import sys
from args_helper import arg_wrapper, _path, model_path
sys.path.append(_path())
sys.setrecursionlimit(10**6) # avoid RecursionError: maximum recursion depth exceeded
from textattack.commands.attack.attack_command import AttackCommand


if __name__ == "__main__":
    args = arg_wrapper()
    args.dataset_from_huggingface = ("snli", None, "train", [1, 2, 0])
    args.recipe = 'textfooler'
    model_name = 'bert-base-uncased-snli'
    args.model = model_name
    args.log_to_csv = '{}/{}-train.csv'.format(model_name, args.recipe)
    args.parallel = False
    args.num_examples = 275076
    attacker = AttackCommand()
    attacker.run(args)
