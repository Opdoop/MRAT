import sys
from args_helper import arg_wrapper, _path, model_path
sys.path.append(_path())
from textattack.commands.attack.attack_command import AttackCommand


if __name__ == "__main__":
    args = arg_wrapper()
    args.dataset_from_file = '../data/snli_test.py'
    args.recipe = 'textfooler'
    model_name = f'bert-base-uncased-snli-mixN-multi'
    args.model_from_file = './models/snli_mixN_multi.py'
    args.parallel = True
    args.log_to_csv = '{}/{}-test-mixN-multi.csv'.format(model_name, args.recipe)
    attacker = AttackCommand()
    attacker.run(args)
