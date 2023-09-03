"""
BSD 3-Clause License

Copyright (c) 2018, NVIDIA Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import os
import sys
import time
import argparse
from shutil import copyfile
from itertools import chain
from collections import OrderedDict

from tqdm import tqdm
import torch
from torch.cuda import amp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

from speech.tps.tps import Handler
from tacotron2.model import load_model
from tacotron2.utils.data_utils import TextMelLoader, TextMelCollate, CustomSampler
from tacotron2.utils.distributed import apply_gradient_allreduce
from tacotron2.modules.optimizers import build_optimizer, build_scheduler, SchedulerTypes
from tacotron2.modules.loss_function import OverallLoss
from tacotron2.hparams import create_hparams
from tacotron2.utils import gradient_adaptive_factor

sys.path.pop(0)

torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def reduce_loss(loss, distributed_run, n_gpus):
    return reduce_tensor(loss.data, n_gpus).item() if distributed_run else loss.item()


def calc_gaf(model, optimizer, loss1, loss2, max_gaf):
    safe_loss = 0. * sum([x.sum() for x in model.parameters()])

    gaf = gradient_adaptive_factor.calc_grad_adapt_factor(
        loss1 + safe_loss, loss2 + safe_loss, model.parameters(), optimizer)
    gaf = min(gaf, max_gaf)

    return gaf


def init_distributed(hparams, n_gpus, rank, group_name):
    print("Initializing Distributed")

    # Initialize distributed communication
    dist.init_process_group(backend=hparams.dist_backend, init_method=hparams.dist_url,
                            world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(hparams, distributed_run=False):
    # Get data, data loaders and collate function ready
    if isinstance(hparams.text_handler_cfg, str):
        text_handler = Handler.from_config(hparams.text_handler_cfg)
        text_handler.out_max_length = None
        assert text_handler.charset.value == hparams.charset
    else:
        text_handler = Handler.from_charset(hparams.charset, data_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"), silent=True)

    trainset = TextMelLoader(text_handler, hparams.training_files, hparams)
    valset = TextMelLoader(text_handler, hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if distributed_run:
        train_sampler = DistributedSampler(trainset)
    else:
        train_sampler = CustomSampler(trainset, hparams.batch_size, hparams.shuffle, hparams.optimize, hparams.len_diff)

    train_loader = DataLoader(trainset, num_workers=1, sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=False, collate_fn=collate_fn)
    return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory, rank):
    root_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, root_path)

    from tacotron2.utils.logger import Tacotron2Logger
    
    sys.path.pop(0)


    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def warm_start_model(checkpoint_path, model, ignore_layers, ignore_mismatched_layers=False):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    pretrained_dict = checkpoint_dict["state_dict"]
    model_dict = model.state_dict()

    # remove extra keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    if ignore_mismatched_layers:
        auto_ignore_layers = []
        for k, v in pretrained_dict.items():
            if v.data.shape != model_dict[k].data.shape:
                auto_ignore_layers.append(k)
        print("Automatically ignored the following pretrained checkpoint keys: ", auto_ignore_layers)
        ignore_layers.extend(auto_ignore_layers)

    if len(ignore_layers) > 0:
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if not any(layer in k for layer in ignore_layers)}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer, lr_scheduler, criterion, restore_lr=True):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))

    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint_dict["state_dict"])
    optimizer.load_state_dict(checkpoint_dict["optimizer"])
    if criterion.mmi_criterion is not None:
        criterion.mmi_criterion.load_state_dict(checkpoint_dict["mi_estimator"])

    iteration = checkpoint_dict["iteration"]

    if not restore_lr:
        base_lr = lr_scheduler.get_last_lr()
        for lr, param_group in zip(base_lr, optimizer.param_groups):
            param_group["lr"] = lr
    else:
        lr_scheduler_params = checkpoint_dict.get("lr_scheduler", None)
        if lr_scheduler_params is not None:
            lr_scheduler.load_state_dict(lr_scheduler_params)

    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))

    return model, optimizer, lr_scheduler, criterion, iteration


def save_checkpoint(model, optimizer, lr_scheduler, criterion, iteration, hparams, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))

    train_dict = {
        "iteration": iteration,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "hparams": hparams.export()
    }

    if criterion.mmi_criterion is not None:
        train_dict["mi_estimator"] = criterion.mmi_criterion.state_dict()

    torch.save(train_dict, filepath)


def validate(model, criterion, valset, iteration, batch_size, collate_fn, logger, distributed_run, rank, n_gpus):
    """Handles all the validation scoring and printing"""
    shuffle = not distributed_run

    losses_dict = OrderedDict({key: [] for key in criterion.list})

    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                shuffle=shuffle, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loader = tqdm(val_loader, desc="Running validation...") if rank == 0 else val_loader
        for i, batch in enumerate(val_loader):
            inputs, alignments, inputs_ctc = model.parse_batch(batch)

            outputs, decoder_outputs = model(inputs)

            losses = criterion(
                outputs, inputs,
                alignments=alignments,
                inputs_ctc=inputs_ctc,
                decoder_outputs=decoder_outputs
            )

            for loss_name, loss_value in losses.items():
                losses_dict[loss_name].append(loss_value)

        num_batches = len(val_loader)
        reduced_losses_dict = {key: [reduce_loss(l, distributed_run, n_gpus) for l in value]
                               for key, value in losses_dict.items()}
        reduced_losses_dict = {key: sum(value) / num_batches for key, value in reduced_losses_dict.items()}

    model.train()
    if rank == 0:
        print("Validation loss {}: {:9f}\n".format(iteration, reduced_losses_dict["overall/loss"]))
        logger.log_validation(reduced_losses_dict, model, inputs, outputs, iteration, alignments)

    return reduced_losses_dict["overall/loss"]


def train(hparams, distributed_run=False, rank=0, n_gpus=None):
    """Training and validation logging results to tensorboard and stdout
    """
    if distributed_run:
        assert n_gpus is not None

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams, distributed_run)
    criterion = OverallLoss(hparams)
    if criterion.mmi_criterion is not None:
        parameters = chain(model.parameters(), criterion.mmi_criterion.parameters())
    else:
        parameters = model.parameters()
    optimizer = build_optimizer(parameters, hparams)
    lr_scheduler = build_scheduler(optimizer, hparams)

    if distributed_run:
        model = apply_gradient_allreduce(model)
    scaler = amp.GradScaler(enabled=hparams.fp16_run)

    logger = prepare_directories_and_logger(hparams.output_dir, hparams.log_dir, rank)
    copyfile(hparams.path, os.path.join(hparams.output_dir, 'hparams.yaml'))
    train_loader, valset, collate_fn = prepare_dataloaders(hparams, distributed_run)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if hparams.checkpoint is not None:
        if hparams.warm_start:
            model = warm_start_model(
                hparams.checkpoint, model, hparams.ignore_layers, hparams.ignore_mismatched_layers)
        else:
            model, optimizer, lr_scheduler, mmi_criterion, iteration = load_checkpoint(
                hparams.checkpoint, model, optimizer, lr_scheduler, criterion, hparams.restore_scheduler_state
            )

            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    model.train()
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()

            model.zero_grad()
            inputs, alignments, inputs_ctc = model.parse_batch(batch)

            with amp.autocast(enabled=hparams.fp16_run):
                outputs, decoder_outputs = model(inputs)

                losses = criterion(
                    outputs, inputs,
                    alignments=alignments,
                    inputs_ctc=inputs_ctc,
                    decoder_outputs=decoder_outputs
                )

            if hparams.use_mmi and hparams.use_gaf and i % gradient_adaptive_factor.UPDATE_GAF_EVERY_N_STEP == 0:
                mi_loss = losses["mi/loss"]
                overall_loss = losses["overall/loss"]

                gaf = calc_gaf(model, optimizer, overall_loss, mi_loss, hparams.max_gaf)

                losses["mi/loss"] = gaf * mi_loss
                losses["overall/loss"] = overall_loss - mi_loss * (1 - gaf)

            reduced_losses = {key: reduce_loss(value, distributed_run, n_gpus) for key, value in losses.items()}
            loss = losses["overall/loss"]

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)

            scaler.step(optimizer)
            scaler.update()

            if rank == 0:
                learning_rate = lr_scheduler.get_last_lr()[0]
                duration = time.perf_counter() - start
                print("Iteration {} ({} epoch): overall loss {:.6f} Grad Norm {:.6f} {:.2f}s/it LR {:.3E}".format(
                    iteration, epoch, reduced_losses["overall/loss"], grad_norm, duration, learning_rate))

                grad_norm = None if torch.isnan(grad_norm) or torch.isinf(grad_norm) else grad_norm
                logger.log_training(reduced_losses, grad_norm, learning_rate, duration, iteration)

            if iteration % hparams.iters_per_checkpoint == 0:
                val_loss = validate(model, criterion, valset, iteration, hparams.batch_size, collate_fn, logger,
                                    distributed_run, rank, n_gpus)
                if rank == 0:
                    checkpoint = os.path.join(
                        hparams.output_dir, "checkpoint_{}".format(iteration))

                    save_checkpoint(model, optimizer, lr_scheduler, criterion, iteration, hparams, checkpoint)

            iteration += 1
            if hparams.lr_scheduler == SchedulerTypes.cyclic:
                lr_scheduler.step()

        if not hparams.lr_scheduler == SchedulerTypes.cyclic:
            if hparams.lr_scheduler == SchedulerTypes.plateau:
                lr_scheduler.step(
                    validate(model, criterion, valset, iteration, hparams.batch_size, collate_fn,
                             logger, distributed_run, rank, n_gpus)
                )
            else:
                lr_scheduler.step()


def main(hparams_path: str = None, distributed_run: bool = False, gpus_ranks: str = False, gpu_idx: int = 0, group_name: str = False):
    if not hparams_path:
        hparams_path = "./data/hparams.yaml"

    hparams = create_hparams(hparams_path)
    hparams.path = hparams_path

    n_gpus = 0
    rank = 0

    if distributed_run:
        assert gpus_ranks
        gpus_ranks = {elem: i for i, elem in enumerate(int(elem) for elem in gpus_ranks.split(","))}
        n_gpus = len(gpus_ranks)
        rank = gpus_ranks[gpu_idx]

        device = "cuda:{}".format(gpu_idx)
    else:
        device = hparams.device.split(":")
        device = device[0] + ":0" if len(device) == 1 else ":".join(device)

    device = torch.device(device)

    if device.type != "cpu":
        assert torch.cuda.is_available()
        print("Use GPU", device)

        torch.cuda.set_device(device)
        if distributed_run:
            init_distributed(hparams, n_gpus, rank, group_name)

        torch.backends.cudnn.enabled = hparams.cudnn_enabled
        torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
    else:
        assert not distributed_run

    hparams.learning_rate = float(hparams.learning_rate)
    hparams.weight_decay = float(hparams.weight_decay)

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    train(hparams, distributed_run=distributed_run, rank=rank, n_gpus=n_gpus)
