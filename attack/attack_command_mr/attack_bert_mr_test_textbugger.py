import sys
from args_helper import arg_wrapper, _path, model_path
sys.path.append(_path())
from textattack.commands.attack.attack_command import AttackCommand


if __name__ == "__main__":
    args = arg_wrapper()
    args.dataset_from_file = '../data/mr_test.py'
    args.recipe = 'textbugger'
    model_name = 'bert-base-uncased-mr'
    args.model = model_path(model_name)
    args.log_to_csv = '{}/{}-test.csv'.format(model_name, args.recipe)
    attacker = AttackCommand()
    attacker.run(args)
