import sys
from args_helper import arg_wrapper, run, _path, _adv_path
sys.path.append(_path())

if __name__ == "__main__":
    args = arg_wrapper()
    args.model = 'bert-base-uncased'
    args.dataset = 'mr'
    args.train_path = '../data/mr/train.txt'
    args.eval_path = '../data/mr/test.txt'

    args.learning_rate = 5e-5
    args.batch_size = 24
    args.early_stopping_epochs = 2
    args.num_train_epochs = 2

    args.adversarial_training = False
    args.mixup_training = True
    args.mix_normal_example = True

    victim_model = f"{args.model}-{args.dataset}"
    recipe = 'multi'
    args.file_paths = [_adv_path(victim_model, 'deepwordbug-train.csv'),
                       _adv_path(victim_model, 'textfooler-train.csv'),
                       _adv_path(victim_model, 'textbugger-train.csv')]

    args.experiment_name = f"{args.model}-{args.dataset}-mixN-{recipe}"
    args.mix_weight = 8

    run(args)