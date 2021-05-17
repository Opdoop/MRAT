import sys
from args_helper import arg_wrapper, _path, model_path
sys.path.append(_path())
sys.setrecursionlimit(10**6) # avoid RecursionError: maximum recursion depth exceeded
from textattack.commands.attack.attack_command import AttackCommand


if __name__ == "__main__":
    args = arg_wrapper()
    args.dataset_from_file = '../data/snli_test.py'
    args.recipe = 'textfooler'
    model_name = 'bert-base-uncased-snli'
    args.model = model_name
    args.log_to_csv = '{}/{}-test.csv'.format(model_name, args.recipe)
    args.parallel = False
    attacker = AttackCommand()
    attacker.run(args)
