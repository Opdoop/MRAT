import sys
from args_helper import arg_wrapper, run, _path, _adv_path
sys.path.append(_path())

if __name__ == "__main__":
    args = arg_wrapper()  # 获取默认需要的全部参数，还没有详细筛选哪些实际有用，TODO，删除无用参数
    # 根据具体实验设置相应参数
    # 模型类型
    args.model = 'bert-base-uncased'
    # 数据集
    args.dataset = 'mr'
    args.train_path = '../data/mr/train.txt'
    args.eval_path = '../data/mr/test.txt'

    # lstm/cnn 与 bert 的学习率和 batch size 设置差别蛮大的，还是手动调一下的好
    args.learning_rate = 2e-5
    args.batch_size = 6
    args.early_stopping_epochs = 3
    args.num_train_epochs = 3

    # 对抗训练的相关设置
    args.adversarial_training = True  # TODO
    # args.save_last = False
    args.mixup_training = False  # 使用使用 mixup 进行 adversarial training， TODO
    args.mix_normal_example = False    # TODO

    # 对抗样本的路径
    victim_model = f"{args.model}-{args.dataset}"
    recipe = 'textbugger'
    recipe_result = 'textbugger-train.csv'
    args.file_paths = [_adv_path(victim_model, recipe_result)]

    # 实验名称
    args.experiment_name = f"{args.model}-{args.dataset}-adv-{recipe}"

    # 跑起来~
    run(args)  # 参数传入主函数