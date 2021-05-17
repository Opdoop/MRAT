"""

TextAttack Command Class for Attack Parralle
---------------------------------------------

A command line parser to run an attack in parralle from user specifications.

"""


from collections import deque
import os
import time

import torch
import tqdm

import textattack

from .attack_args_helpers import (
    parse_attack_from_args,
    parse_dataset_from_args,
    parse_logger_from_args,
)

logger = textattack.shared.logger


def set_env_variables(gpu_id):
    # Disable tensorflow logs, except in the case of an error.
    if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Set sharing strategy to file_system to avoid file descriptor leaks
    torch.multiprocessing.set_sharing_strategy("file_system")

    # Only use one GPU, if we have one.
    # For Tensorflow
    # TODO: Using USE with `--parallel` raises similar issue as https://github.com/tensorflow/tensorflow/issues/38518#
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # For PyTorch
    torch.cuda.set_device(gpu_id)

    # Fix TensorFlow GPU memory growth
    try:
        import tensorflow as tf

        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                gpu = gpus[gpu_id]
                tf.config.experimental.set_visible_devices(gpu, "GPU")
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
    except ModuleNotFoundError:
        pass


def attack_from_queue(args, in_queue, out_queue):
    num_gpus = torch.cuda.device_count()
    gpu_id = (torch.multiprocessing.current_process()._identity[0] - 2) % num_gpus
    set_env_variables(gpu_id)
    textattack.shared.utils.set_seed(args.random_seed)
    attack = parse_attack_from_args(args)
    if gpu_id == 0:
        print(attack, "\n")
    while not in_queue.empty():
        try:
            i, text, output = in_queue.get()
            results_gen = attack.attack_dataset([(text, output)])
            result = next(results_gen)
            out_queue.put((i, result))
        except Exception as e:
            out_queue.put(e)
            exit()


def run(args, checkpoint=None):
    # 入口方法，传入参数
    pytorch_multiprocessing_workaround()  # 什么修复 bug 的方法

    dataset = parse_dataset_from_args(args)  # 从参数中获取数据集，这里会更改 args.num_examples，从 -1 改为真实数据集大小
    if args.num_examples == -1:
        num_total_examples = len(dataset)  # args.num_examples  # 获取样本数
    else:
        num_total_examples = args.num_examples

    if args.checkpoint_resume:  # 若从存档点继续
        num_remaining_attacks = checkpoint.num_remaining_attacks  # 读取剩余的待攻击样本个数
        worklist = checkpoint.worklist # 获取工作队列
        worklist_tail = checkpoint.worklist_tail  # 获取当前对尾
        logger.info(  # 打印日志
            "Recovered from checkpoint previously saved at {}".format(
                checkpoint.datetime
            )
        )
        print(checkpoint, "\n")
    else:  # 新建
        num_remaining_attacks = num_total_examples  # 总样本数
        worklist = deque(range(0, num_total_examples)) # 双向队列，记录当前位置
        worklist_tail = worklist[-1]  # 获取队尾

    # This makes `args` a namespace that's sharable between processes.
    # We could do the same thing with the model, but it's actually faster
    # to let each thread have their own copy of the model.
    args = torch.multiprocessing.Manager().Namespace(**vars(args))  # 使得 args 全局共享

    if args.checkpoint_resume:  # 获取 log_manager ，用于记录日志，进行日志打印，保存等操作
        attack_log_manager = checkpoint.log_manager
    else:
        attack_log_manager = parse_logger_from_args(args)  # 创建

    # We reserve the first GPU for coordinating workers. # 使用第一个 GPU 来调度多进程
    num_gpus = torch.cuda.device_count()  # 获取 GPU 数量

    textattack.shared.logger.info(f"Running on {num_gpus} GPUs")
    start_time = time.time()  # 开始计时

    if args.interactive:  # 多线程不支持从 stdin 交互式输入
        raise RuntimeError("Cannot run in parallel if --interactive set")

    in_queue = torch.multiprocessing.Queue()  # 输入队列，可在线程间交互
    out_queue = torch.multiprocessing.Queue()  # # 输出队列，可在线程间交互
    # Add stuff to queue.
    missing_datapoints = set()  # 缺失样本集合
    for i in worklist:  # 遍历所有工作列表
        try:
            text, output = dataset[i]  # 获取输入
            in_queue.put((i, text, output))  # 放入输入队列
        except IndexError:  # 列表遍历到的下标 i 非法
            missing_datapoints.add(i)  # 下标加入缺失样本集合

    # if our dataset is shorter than the number of samples chosen, remove the
    # out-of-bounds indices from the dataset
    for i in missing_datapoints:  # 把缺失的样本的下标从工作列表中清除
        worklist.remove(i)

    # Start workers.
    torch.multiprocessing.Pool(num_gpus, attack_from_queue, (args, in_queue, out_queue))  # 创建多线程池，Pool(线程数，调用函数，（调用函数传入的参数））
    # Log results asynchronously and update progress bar.
    if args.checkpoint_resume:  # 如果从存档点开始， 读取已有结果
        num_results = checkpoint.results_count  # 已攻击样本数
        num_failures = checkpoint.num_failed_attacks  # 已失败个数
        num_successes = checkpoint.num_successful_attacks  # 已成功个数
    else:  # 初始化，从 0 开始
        num_results = 0
        num_failures = 0
        num_successes = 0
    pbar = tqdm.tqdm(total=num_remaining_attacks, smoothing=0)  # 创建进度条
    while worklist:  # 循环所有的样本
        result = out_queue.get(block=True)  # 获取结果，wtf？魔法？
        if isinstance(result, Exception):  # 如果 out_queue 返回的 result 类型是 Exception
            raise result  # 接着报错
        idx, result = result  # 获取到正常结果
        attack_log_manager.log_result(result)  # logger 记录结果
        worklist.remove(idx)  # 将这个样本对应的下标从工作列表移除
        if (not args.attack_n) or (
            not isinstance(result, textattack.attack_results.SkippedAttackResult)
        ):  # 如果没有设置攻击次数 attack_n 用于 iterative 方式，或攻击结果不是跳过，skip（模型原本就分错的）
            pbar.update()  # 更新进度条
            num_results += 1  # 总攻击数量 + 1

            if (
                type(result) == textattack.attack_results.SuccessfulAttackResult  # 如果结果为成功
                or type(result) == textattack.attack_results.MaximizedAttackResult
            ):
                num_successes += 1   # 成功个数 + 1
            if type(result) == textattack.attack_results.FailedAttackResult:  # 如果结果为失败
                num_failures += 1  # 失败个数 +1
            pbar.set_description(  # 更新进度条
                "[Succeeded / Failed / Total] {} / {} / {}".format(
                    num_successes, num_failures, num_results
                )
            )
        else:
            # 设置了 attack_n 个数，但继续攻击
            # worklist_tail keeps track of highest idx that has been part of worklist
            # Used to get the next dataset element when attacking with `attack_n` = True.
            worklist_tail += 1
            try:
                text, output = dataset[worklist_tail]
                worklist.append(worklist_tail)
                in_queue.put((worklist_tail, text, output))
            except IndexError:
                raise IndexError(
                    "Tried adding to worklist, but ran out of datapoints. Size of data is {} but tried to access index {}".format(
                        len(dataset), worklist_tail
                    )
                )

        if (  # 如果要进行存档
            args.checkpoint_interval
            and len(attack_log_manager.results) % args.checkpoint_interval == 0
        ):
            new_checkpoint = textattack.shared.Checkpoint(
                args, attack_log_manager, worklist, worklist_tail
            )
            new_checkpoint.save()
            attack_log_manager.flush()

    pbar.close()  # 关闭进度条
    print()
    # Enable summary stdout.
    if args.disable_stdout:
        attack_log_manager.enable_stdout()
    attack_log_manager.log_summary()  # 输出日志
    attack_log_manager.write_summary()  # 保存日志
    attack_log_manager.flush()  # 输出+清空
    print()  # 空行

    textattack.shared.logger.info(f"Attack time: {time.time() - start_time}s")  # 输出时间

    return attack_log_manager.results


def pytorch_multiprocessing_workaround():
    # This is a fix for a known bug
    try:
        torch.multiprocessing.set_start_method("spawn")
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        pass
