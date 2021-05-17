import sys
from args_helper import arg_wrapper, run, _path, _adv_path
sys.path.append(_path())

if __name__ == "__main__":
    args = arg_wrapper()
    args.model = 'bert-base-uncased'
    args.dataset = 'snli'
    args.train_path = '../data/snli/snli_1.0_train.txt'
    args.eval_path = '../data/snli/snli_1.0_test.txt'

    args.learning_rate = 2e-05
    args.batch_size = 24
    args.early_stopping_epochs = 5
    args.num_train_epochs = 5

    args.adversarial_training = False
    args.save_last = False
    args.mixup_training = True
    args.mix_normal_example = False

    victim_model = f"{args.model}-{args.dataset}"
    recipe = 'multi'
    args.file_paths = [_adv_path(victim_model, 'deepwordbug-train.csv'),
                       _adv_path(victim_model, 'textfooler-train.csv'),
                       _adv_path(victim_model, 'textbugger-train.csv')]
    args.mix_weight = 8

    args.experiment_name = f"{args.model}-{args.dataset}-mix-{recipe}"

    run(args)