import sys
from args_helper import arg_wrapper, run, _path
sys.path.append(_path())

if __name__ == "__main__":
    args = arg_wrapper()
    args.model = 'bert-base-uncased'
    args.dataset = 'mr'
    args.train_path = '../data/mr/train.txt'
    args.eval_path = '../data/mr/test.txt'

    args.learning_rate = 5e-5
    args.batch_size = 16
    args.early_stopping_epochs = 3
    args.num_train_epochs = 3

    args.adversarial_training = False
    args.mixup_training = False
    args.mix_normal_example = False
    args.save_last = False

    args.experiment_name = f"{args.model}-{args.dataset}"

    run(args)