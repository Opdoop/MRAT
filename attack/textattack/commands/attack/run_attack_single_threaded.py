"""

TextAttack Command Class for Attack Single Threaded
-----------------------------------------------------

A command line parser to run an attack in single thread from user specifications.

"""

from collections import deque
import os
import time

import tqdm

import textattack

from .attack_args_helpers import (
    parse_attack_from_args,
    parse_dataset_from_args,
    parse_logger_from_args,
)

logger = textattack.shared.logger


def run(args, checkpoint=None):
    # 主逻辑函数
    # Only use one GPU, if we have one.
    # TODO: Running Universal Sentence Encoder uses multiple GPUs
    # 设置 GPU
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Disable tensorflow logs, except in the case of an error.
    if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # 尝试设置 TensorFlow GPU 内从动态增长
    try:
        # Fix TensorFlow GPU memory growth
        import tensorflow as tf

        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:  # 如果获取到了 GPU
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:  # 对于所有 GPU
                    tf.config.experimental.set_memory_growth(gpu, True)  # 设置动态增长为 True
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)  # 如果无法设置动态增长，报错，提示必须在初始化时指定 GPU 的动态增长
    except ModuleNotFoundError:
        pass

    if args.checkpoint_resume:
        # 如果从 checkpoint 处继续
        num_remaining_attacks = checkpoint.num_remaining_attacks  # 获取剩余需要攻击的数量
        worklist = checkpoint.worklist  # 获取所有的需要攻击的样本的顺序队列
        worklist_tail = checkpoint.worklist_tail  # 获取当前以遍历到的位置

        logger.info(
            "Recovered from checkpoint previously saved at {}".format(
                checkpoint.datetime
            )
        )
        print(checkpoint, "\n")
    else:
        # 如果从新开始
        if not args.interactive:  # 如果不是交互式的，交互式指样本为用户从 cli 输入
            dataset = parse_dataset_from_args(args)  # 从传入的参数中读取数据集
        if args.num_examples == -1:
            num_remaining_attacks = len(dataset)  # args.num_examples  # 获取样本数量
            worklist = deque(range(0, len(
                dataset)))  # args.num_examples))  # 双向队列，长度为 0 到样本数，比 list append 复杂度低 list O(n), deque O(1)
            worklist_tail = worklist[-1]  # 当前的序列尾
        else:
            num_total_examples = args.num_examples
            num_remaining_attacks = num_total_examples   # 获取样本数量
            worklist = deque(range(0, num_total_examples)) # 双向队列，长度为 0 到样本数，比 list append 复杂度低 list O(n), deque O(1)
            worklist_tail = worklist[-1]  # 当前的序列尾

    # 记录开始时间
    start_time = time.time()

    # Attack
    # 获取 Attack model
    attack = parse_attack_from_args(args)
    print(attack, "\n")

    # Logger
    # 获取 Logger
    if args.checkpoint_resume:
        # 如果从 checkpoint 继续
        attack_log_manager = checkpoint.log_manager  # 读取 checkpoint 的 logger
    else:
        # 如果从新开始
        attack_log_manager = parse_logger_from_args(args)  # 从参数读取 logger

    # 截止载入时间
    load_time = time.time()
    textattack.shared.logger.info(f"Load time: {load_time - start_time}s")  # 打印载入耗时

    # 如果是交互式的攻击
    if args.interactive:
        print("Running in interactive mode")
        print("----------------------------")

        while True:
            print('Enter a sentence to attack or "q" to quit:')
            text = input()

            if text == "q":
                break

            if not text:
                continue

            print("Attacking...")

            # 构造攻击
            attacked_text = textattack.shared.attacked_text.AttackedText(text)  # 用户输入的样本，构造为攻击需要的输入格式
            initial_result = attack.goal_function.get_output(attacked_text)   # 使用 goal_function 得到这个样本的预期结果（是 pseudo-label 吗？）
            result = next(attack.attack_dataset([(text, initial_result)]))  # 获取攻击得到的结果
            print(result.__str__(color_method="ansi") + "\n")  # 打印结果

    else:
        # Not interactive? Use default dataset.
        # 如果不是交互式的，使用默认的数据集
        pbar = tqdm.tqdm(total=num_remaining_attacks, smoothing=0)  # 初始化一个进度条
        if args.checkpoint_resume:  # 如果是从 checkpoint 继续
            num_results = checkpoint.results_count  # 读取已有结果
            num_failures = checkpoint.num_failed_attacks  # 读取已经失败的数量
            num_successes = checkpoint.num_successful_attacks  # 读取已成功的数量
        else:  # 否则，从 0 初始化
            num_results = 0
            num_failures = 0
            num_successes = 0

        # 对 dataset 进行循环
        for result in attack.attack_dataset(dataset, indices=worklist):  # 上面创建的 deque: worklist 作为下标
            attack_log_manager.log_result(result)  # 对一次攻击的 result 输出

            if not args.disable_stdout:  # 如果不输出到屏幕
                print("\n")  # 输出空行
            if (not args.attack_n) or ( # attack_n 为 false
                    not isinstance(result, textattack.attack_results.SkippedAttackResult)  # 或得到的 attack 样本的结果是 skip
            ):
                pbar.update(1)  # 进度条更新 1
            else:
                # 使用 tail 来跟踪当前位置
                # worklist_tail keeps track of highest idx that has been part of worklist
                # Used to get the next dataset element when attacking with `attack_n` = True.
                worklist_tail += 1
                worklist.append(worklist_tail)  # 将 tail 拼接在 list 里？  why？

            num_results += 1  #以获取的样本数 +1

            if (  # 如果攻击样本的结果是成功
                    type(result) == textattack.attack_results.SuccessfulAttackResult
                    or type(result) == textattack.attack_results.MaximizedAttackResult  # Maximized 和 Successful 有什么区别，不清楚
            ):
                num_successes += 1  # 成功数量 +1
            # 如果攻击失败
            if type(result) == textattack.attack_results.FailedAttackResult:
                num_failures += 1  # 失败数量 +1
            # 进度提显式当前结果
            pbar.set_description(
                "[Succeeded / Failed / Total] {} / {} / {}".format(
                    num_successes, num_failures, num_results
                )
            )

            if (
                    args.checkpoint_interval  # 如果设置了没 N 轮保存一次 checkpoint
                    and len(attack_log_manager.results) % args.checkpoint_interval == 0  # 当前结果轮为保存 checkpoint 的轮数
            ):
                # 传入参数，logger，worklist 和 worklist_tail，得到新的 checkpoint
                new_checkpoint = textattack.shared.Checkpoint(
                    args, attack_log_manager, worklist, worklist_tail
                )
                # 保存 checkpoint
                new_checkpoint.save()
                attack_log_manager.flush()  # 清空 logger ?
        # 循环结束，攻击结束，关闭进度条
        pbar.close()
        print()
        # Enable summary stdout
        # 攻击结束，单条样本是否输出到屏幕，都将 summary 输出到屏幕
        if args.disable_stdout:
            attack_log_manager.enable_stdout()
        attack_log_manager.log_summary()
        attack_log_manager.write_summary()
        attack_log_manager.flush()
        print()
        # finish_time = time.time()
        # 输出攻击耗时
        textattack.shared.logger.info(f"Attack time: {time.time() - load_time}s")

        # 返回攻击得到的结果，成功样本与失败样本都返回
        return attack_log_manager.results
