import json
import logging
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
import tqdm
import transformers
from torch.nn import functional as F

import train.shared

from .perturbed_helper import PerturbedDataset

from .train_args_helpers import (
    dataset_from_args,
    model_from_args,
    write_readme,
    dataset_from_local,
)

device = train.shared.utils.device
logger = train.shared.logger


def _save_args(args, save_path):
    """Dump args dictionary to a json.
    :param: args. Dictionary of arguments to save.
    :save_path: Path to json file to write args to.
    """
    final_args_dict = {k: v for k, v in vars(args).items() if _is_writable_type(v)}
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(final_args_dict, indent=2) + "\n")


def _get_sample_count(*lsts):
    """Get sample count of a dataset.
    :param *lsts: variable number of lists.
    :return: sample count of this dataset, if all lists match, else None.
    """
    if all(len(lst) == len(lsts[0]) for lst in lsts):
        sample_count = len(lsts[0])
    else:
        sample_count = None
    return sample_count


def _random_shuffle(*lsts):
    """Randomly shuffle a dataset. Applies the same permutation to each list
    (to preserve mapping between inputs and targets).

    :param *lsts: variable number of lists to shuffle.
    :return: shuffled lsts.
    """
    permutation = np.random.permutation(len(lsts[0]))
    shuffled = []
    for lst in lsts:
        shuffled.append((np.array(lst)[permutation]).tolist())
    return tuple(shuffled)


def _train_val_split(*lsts, split_val=0.2):
    """Split dataset into training and validation sets.
    :param *lsts: variable number of lists that make up a dataset (e.g. text, labels)
    :param split_val: float [0., 1.). Fraction of the dataset to reserve for evaluation.
    :return: (train split of list for list in lsts), (val split of list for list in lsts)
    """
    sample_count = _get_sample_count(*lsts)
    if not sample_count:
        raise Exception(
            "Batch Axis inconsistent. All input arrays must have first axis of equal length."
        )
    lsts = _random_shuffle(*lsts)
    split_idx = math.floor(sample_count * split_val)
    train_set = [lst[split_idx:] for lst in lsts]
    val_set = [lst[:split_idx] for lst in lsts]
    if len(train_set) == 1 and len(val_set) == 1:
        train_set = train_set[0]
        val_set = val_set[0]
    return train_set, val_set


def _save_model_checkpoint(model, output_dir, global_step):
    """Save model checkpoint to disk.
    checkpoint
    :param model: Model to save (pytorch)
    :param output_dir: Path to model save dir.
    :param global_step: Current global training step #. Used in ckpt filename.
    """
    # Save model checkpoint
    output_dir = os.path.join(output_dir, "checkpoint-{}".format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)


def _save_model(model, output_dir, weights_name, config_name):
    """Save model to disk.
    :param model: Model to save (pytorch)
    :param output_dir: Path to model save dir.
    :param weights_name: filename for model parameters.
    :param config_name: filename for config.
    """
    model_to_save = model.module if hasattr(model, "module") else model

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(output_dir, weights_name)
    output_config_file = os.path.join(output_dir, config_name)

    # state_dict()
    torch.save(model_to_save.state_dict(), output_model_file)
    try:
        model_to_save.config.to_json_file(output_config_file)
    except AttributeError:
        # no config
        pass


def _get_eval_score(model, eval_dataloader, normal=False):
    """Measure performance of a model on the evaluation set.
    :param model: Model to test.
    :param eval_dataloader: a torch DataLoader that iterates through the eval set.

    :return: pearson correlation, if do_regression==True, else classification accuracy [0., 1.]
    """
    model.eval()
    correct = 0
    logits = []
    labels = []
    for input_ids, batch_labels in eval_dataloader:
        batch_labels = batch_labels.to(device)
        if isinstance(input_ids, dict):
            ## dataloader collates dict backwards. This is a workaround to get
            # ids in the right shape for HuggingFace models
            input_ids = {k: torch.stack(v).T.to(device) for k, v in input_ids.items()}
            with torch.no_grad():
                if normal:
                    batch_logits = model(**input_ids)[0]
                else:
                    batch_logits = model(input_ids)
        else:
            input_ids = input_ids.to(device)
            with torch.no_grad():
                batch_logits = model(input_ids)

        logits.extend(batch_logits.cpu().squeeze().tolist())
        labels.extend(batch_labels)

    model.train()

    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum()
    return float(correct) / len(labels)


def _make_directories(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def _is_writable_type(obj):
    for ok_type in [bool, int, str, float]:
        if isinstance(obj, ok_type):
            return True
    return False


def batch_encode(tokenizer, text_list):
    if hasattr(tokenizer, "batch_encode"):
        return tokenizer.batch_encode(text_list)
    else:
        return [tokenizer.encode(text_input) for text_input in text_list]

def _mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def _batch_encoder(tokenizer, text):
    '''
    Large text list cause process killed. Orderly process
    :param tokenizer:
    :param text:
    :return:
    '''
    text_ids = []
    batch_number = len(text)//10000
    start, end = 0, 0
    for i in range(batch_number):
        start = i * 10000
        end = (i+1) * 10000
        text_ids.extend(batch_encode(tokenizer, text[start:end]))
    text_ids.extend(batch_encode(tokenizer, text[end:]))
    return text_ids

def _make_dataloader(tokenizer, text, labels, batch_size):
    """Create torch DataLoader from list of input text and labels.
    :param tokenizer: Tokenizer to use for this text.
    :param text: list of input text.
    :param labels: list of corresponding labels.
    :param batch_size: batch size (int).
    :return: torch DataLoader for this training set.
    """

    text_ids = _batch_encoder(tokenizer, text)
    input_ids = np.array(text_ids)
    labels = np.array(labels)
    data = list((ids, label) for ids, label in zip(input_ids, labels))
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader



def train_model(args):
    train.shared.utils.set_seed(args.random_seed)

    _make_directories(args.output_dir)

    num_gpus = torch.cuda.device_count()

    # Save logger writes to file
    log_txt_path = os.path.join(args.output_dir, "log.txt")
    fh = logging.FileHandler(log_txt_path)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.info(f"Writing logs to {log_txt_path}.")

    # Get list of text and list of label (integers) from disk.
    train_text, train_labels, eval_text, eval_labels = dataset_from_local(args)

    if args.pct_dataset < 1.0:
        logger.info(f"Using {args.pct_dataset*100}% of the training set")
        (train_text, train_labels), _ = _train_val_split(
            train_text, train_labels, split_val=1.0 - args.pct_dataset
        )
    train_examples_len = len(train_text)

    label_set = set(train_labels)
    args.num_labels = len(label_set)
    logger.info(f"Loaded dataset. Found: {args.num_labels} labels: {sorted(label_set)}")

    if len(train_labels) != len(train_text):
        raise ValueError(
            f"Number of train examples ({len(train_text)}) does not match number of labels ({len(train_labels)})"
        )
    if len(eval_labels) != len(eval_text):
        raise ValueError(
            f"Number of teste xamples ({len(eval_text)}) does not match number of labels ({len(eval_labels)})"
        )

    model_wrapper = model_from_args(args, args.num_labels)
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer

    # multi-gpu training
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
        logger.info("Using torch.nn.DataParallel.")
    logger.info(f"Training model across {num_gpus} GPUs")

    num_train_optimization_steps = (
        int(train_examples_len / args.batch_size / args.grad_accum_steps)
        * args.num_train_epochs
    )

    if args.model == "lstm" or args.model == "cnn":
        def need_grad(x):
            return x.requires_grad
        optimizer = torch.optim.Adam(
            filter(need_grad, model.parameters()), lr=args.learning_rate
        )
        scheduler = None
    else:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = transformers.optimization.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate
        )

        scheduler = transformers.optimization.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_proportion,
            num_training_steps=num_train_optimization_steps,
        )

    # Start Tensorboard and log hyperparams.

    from torch.utils.tensorboard import SummaryWriter

    tb_writer = SummaryWriter(args.output_dir)

    # Use Weights & Biases, if enabled.

    if args.enable_wandb:
        global wandb
        wandb = train.shared.utils.LazyLoader("wandb", globals(), "wandb")

        wandb.init(sync_tensorboard=True)

    # Save original args to file

    args_save_path = os.path.join(args.output_dir, "train_args.json")
    _save_args(args, args_save_path)
    logger.info(f"Wrote original training args to {args_save_path}.")

    tb_writer.add_hparams(
        {k: v for k, v in vars(args).items() if _is_writable_type(v)}, {}
    )

    eval_dataloader = _make_dataloader(
        tokenizer, eval_text, eval_labels, args.batch_size
    )

    train_dataloader = _make_dataloader(
        tokenizer, train_text, train_labels, args.batch_size
    )

    global_step = 0
    tr_loss = 0

    model.train()
    args.best_eval_score = 0
    args.best_eval_score_epoch = 0
    args.epochs_since_best_eval_score = 0

    def loss_backward(loss):
        if num_gpus > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training，
        if args.grad_accum_steps > 1:
            loss = loss / args.grad_accum_steps
        loss.backward()
        return loss

    loss_fct = torch.nn.CrossEntropyLoss()


    if args.adversarial_training:
        logger.info(f"Read perturbed dataset from file {args.file_paths}")
        adv_dataset = PerturbedDataset(args.file_paths, tokenizer)
        perturbed_text, perturbed_label = adv_dataset.perturbed_string()
        train_dataloader = _make_dataloader(
            tokenizer, train_text+perturbed_text, train_labels+perturbed_label, args.batch_size
        )
        train_examples_len = len(train_text+adv_dataset.perturbed_list)
    if args.mixup_training:
        logger.info(f"Read perturbed dataset from file {args.file_paths}")
        adv_dataset = PerturbedDataset(args.file_paths, tokenizer)
        adv_dataloader = DataLoader(adv_dataset, shuffle=True, batch_size=args.batch_size)
        train_dataloader_b = _make_dataloader(
            tokenizer, train_text, train_labels, args.batch_size
        )
        train_examples_len = len(adv_dataset.perturbed_list)

    # Start training
    logger.info("***** Running training *****")
    logger.info(f"\tNum examples = {train_examples_len}")
    logger.info(f"\tBatch size = {args.batch_size}")
    logger.info(f"\tMax sequence length = {args.max_length}")
    logger.info(f"\tNum steps = {num_train_optimization_steps}")
    logger.info(f"\tNum epochs = {args.num_train_epochs}")
    logger.info(f"\tLearning rate = {args.learning_rate}")

    mix_weight = args.mix_weight if args.mix_weight else 6
    for epoch in tqdm.trange(
        int(args.num_train_epochs), desc="Epoch", position=0, leave=True
    ):
        if args.mixup_training:
            prog_bar = tqdm.tqdm(adv_dataloader, desc="Iteration", position=0, leave=True)

            # Use these variables to track training accuracy during classification.
            correct_predictions = 0
            total_predictions = 0
            # run step
            for step, batch in enumerate(zip(train_dataloader, prog_bar, train_dataloader_b)):
                normal_batch, adv_batch, normal_batch_b = batch

                origin_ids_a, perturb_ids, labels_a = adv_batch
                origin_ids_a = {k: torch.stack(v).T.to(device) for k, v in origin_ids_a.items()} if isinstance(origin_ids_a, dict) else origin_ids_a.to(device)
                labels_a = labels_a.to(device)
                perturb_ids = {k: torch.stack(v).T.to(device) for k, v in perturb_ids.items()} if isinstance(perturb_ids, dict) else perturb_ids.to(device)

                logits = model(origin_ids_a)
                loss = loss_fct(logits, labels_a)
                m = np.float32(np.random.beta(1, 1))

                if args.regularized_adv_example:
                    if args.adv_mixup:
                        logits_mix = model(_input=origin_ids_a, perturbed_input=perturb_ids, mix_ratio=m)
                        prob_logits, prob_mix = F.softmax(logits, dim=1), F.softmax(logits_mix, dim=1)
                        p_mixture = torch.clamp((prob_logits + prob_mix) / 2., 1e-7, 1).log()
                        loss +=  mix_weight * (F.kl_div(p_mixture, prob_logits, reduction='batchmean') +
                                     F.kl_div(p_mixture, prob_mix, reduction='batchmean')) / 2.
                    else:
                        logits_mix = model(perturb_ids)
                        prob_logits, prob_mix = F.softmax(logits, dim=1), F.softmax(logits_mix, dim=1)
                        p_mixture = torch.clamp((prob_logits + prob_mix) / 2., 1e-7, 1).log()
                        loss +=  mix_weight * (F.kl_div(p_mixture, prob_logits, reduction='batchmean') +
                                     F.kl_div(p_mixture, prob_mix, reduction='batchmean')) / 2.


                if args.mix_normal_example:
                    origin_ids_n, labels_n = normal_batch
                    origin_ids_b, labels_b = normal_batch_b
                    labels_n = labels_n.to(device)
                    origin_ids_n = {k: torch.stack(v).T.to(device) for k, v in origin_ids_n.items()} if isinstance(origin_ids_n, dict) else origin_ids_n.to(device)
                    labels_b = labels_b.to(device)
                    origin_ids_b = {k: torch.stack(v).T.to(device) for k, v in origin_ids_b.items()}if isinstance(origin_ids_b, dict) else origin_ids_b.to(device)
                    logits_normal_mix = model(_input=origin_ids_n, perturbed_input=origin_ids_b, mix_ratio=m)
                    loss += _mixup_criterion(loss_fct, logits_normal_mix, labels_n, labels_b, m)

                pred_labels = logits.argmax(dim=-1)
                correct_predictions += (pred_labels == labels_a).sum().item()
                total_predictions += len(pred_labels)

                loss = loss_backward(loss)
                tr_loss += loss.item()

                if global_step % args.tb_writer_step == 0:
                    tb_writer.add_scalar("loss", loss.item(), global_step)

                    if scheduler is not None:
                        tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    else:
                        tb_writer.add_scalar("lr", args.learning_rate, global_step)

                if global_step > 0:
                    prog_bar.set_description(f"Epoch {epoch} Loss {tr_loss / global_step}")
                if (step + 1) % args.grad_accum_steps == 0:
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                # Save model checkpoint to file.

                if (
                        global_step > 0
                        and (args.checkpoint_steps > 0)
                        and (global_step % args.checkpoint_steps) == 0
                ):
                    _save_model_checkpoint(model, args.output_dir, global_step)

                # Inc step counter.
                global_step += 1
        else:
            # normal training or adversarial training
            prog_bar = tqdm.tqdm(train_dataloader, desc="Iteration", position=0, leave=True)

            # Use these variables to track training accuracy during classification.
            correct_predictions = 0
            total_predictions = 0

            for step, batch in enumerate(prog_bar):
                input_ids, labels = batch

                labels = labels.to(device)

                if isinstance(input_ids, dict):
                    ## dataloader collates dict backwards. This is a workaround to get
                    # ids in the right shape for HuggingFace models
                    input_ids = {
                        k: torch.stack(v).T.to(device) for k, v in input_ids.items()
                    }
                    if args.adversarial_training:
                        logits = model(input_ids)
                    else:
                        logits = model(**input_ids)[0]
                else:
                    input_ids = input_ids.to(device)
                    logits = model(input_ids)

                loss = loss_fct(logits, labels)

                pred_labels = logits.argmax(dim=-1)
                correct_predictions += (pred_labels == labels).sum().item()
                total_predictions += len(pred_labels)

                loss = loss_backward(loss)
                tr_loss += loss.item()

                if global_step % args.tb_writer_step == 0:
                    tb_writer.add_scalar("loss", loss.item(), global_step)
                    if scheduler is not None:
                        tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    else:
                        tb_writer.add_scalar("lr", args.learning_rate, global_step)

                if global_step > 0:
                    prog_bar.set_description(f"Epoch {epoch} Loss {tr_loss/global_step}")
                if (step + 1) % args.grad_accum_steps == 0:
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                # Save model checkpoint to file.
                if (
                    global_step > 0
                    and (args.checkpoint_steps > 0)
                    and (global_step % args.checkpoint_steps) == 0
                ):
                    _save_model_checkpoint(model, args.output_dir, global_step)

                # Inc step counter.
                global_step += 1

        # Print training accuracy, if we're tracking it.
        if total_predictions > 0:
            train_acc = correct_predictions / total_predictions
            logger.info(f"Train accuracy: {train_acc*100}%")
            tb_writer.add_scalar("epoch_train_score", train_acc, epoch)

        # Check accuracy after each epoch.
        # skip args.num_clean_epochs during adversarial training
        if (epoch >= args.num_clean_epochs):
            if args.adversarial_training:
                eval_score = _get_eval_score(model, eval_dataloader)
            else:
                eval_score = _get_eval_score(model, eval_dataloader, normal=True)
            tb_writer.add_scalar("epoch_eval_score", eval_score, epoch)

            if args.checkpoint_every_epoch:
                _save_model_checkpoint(model, args.output_dir, args.global_step)

            logger.info(
                f"Eval accuracy: {eval_score*100}%"
            )
            if eval_score > args.best_eval_score:
                args.best_eval_score = eval_score
                args.best_eval_score_epoch = epoch
                args.epochs_since_best_eval_score = 0
                _save_model(model, args.output_dir, args.weights_name, args.config_name)
                logger.info(f"Best acc found. Saved model to {args.output_dir}.")
                _save_args(args, args_save_path)
                logger.info(f"Saved updated args to {args_save_path}")
            else:
                args.epochs_since_best_eval_score += 1
                if (args.early_stopping_epochs > 0) and (
                    args.epochs_since_best_eval_score > args.early_stopping_epochs
                ):
                    logger.info(
                        f"Stopping early since it's been {args.early_stopping_epochs} epochs since validation acc increased"
                    )
                    break

    if args.save_last:
        _save_model(model, args.output_dir, args.weights_name, args.config_name)

    # read the saved model and report its eval performance
    logger.info("Finished training. Re-loading and evaluating model from disk.")
    model_wrapper = model_from_args(args, args.num_labels)
    model = model_wrapper.model
    model.load_state_dict(torch.load(os.path.join(args.output_dir, args.weights_name)))
    if args.adversarial_training or args.mixup_training:
        eval_score = _get_eval_score(model, eval_dataloader)
    else:
        eval_score = _get_eval_score(model, eval_dataloader, normal=True)

    if args.save_last:
        args.best_eval_score = eval_score
        args.best_eval_score_epoch = epoch
    logger.info(
        f"Saved model accuracy: {eval_score*100}%"
    )

    # end of training, save tokenizer
    try:
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Saved tokenizer {tokenizer} to {args.output_dir}.")
    except AttributeError:
        logger.warn(
            f"Error: could not save tokenizer {tokenizer} to {args.output_dir}."
        )

    # Save a little readme with model info
    write_readme(args, args.best_eval_score, args.best_eval_score_epoch)

    _save_args(args, args_save_path)
    tb_writer.close()
    logger.info(f"Wrote final training args to {args_save_path}.")
