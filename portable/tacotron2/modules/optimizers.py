import warnings
from enum import Enum

from torch import optim
import torch_optimizer


class OptimizersTypes(str, Enum):
    sgd = "sgd"
    yogi = "yogi"
    adam = "adam"
    radam = "radam"
    diffgrad = "diffgrad"
    novograd = "novograd"
    adabound = "adabound"


optimizers = {
    OptimizersTypes.sgd: optim.SGD,
    OptimizersTypes.yogi: torch_optimizer.Yogi,
    OptimizersTypes.adam: optim.Adam,
    OptimizersTypes.radam: torch_optimizer.RAdam,
    OptimizersTypes.diffgrad: torch_optimizer.DiffGrad,
    OptimizersTypes.novograd: torch_optimizer.NovoGrad,
    OptimizersTypes.adabound: torch_optimizer.AdaBound
}

optimizers_options = {
    OptimizersTypes.sgd: ["momentum", "dampening", "nesterov"],
    OptimizersTypes.yogi: ["betas", "eps", "initial_accumulator"],
    OptimizersTypes.adam: ["betas", "eps", "amsgrad"],
    OptimizersTypes.radam: ["betas", "eps"],
    OptimizersTypes.diffgrad: ["betas", "eps"],
    OptimizersTypes.novograd: ["betas", "eps", "grad_averaging", "amsgrad"],
    OptimizersTypes.adabound: ["betas", "eps", "final_lr", "gamma", "amsbound"]
}


def build_optimizer(parameters, hparams):
    optimizer_type = OptimizersTypes[hparams.optimizer]
    optimizer_opts = {} if hparams.optim_options is None else hparams.optim_options

    if optimizer_type in OptimizersTypes:
        if not all(arg in optimizers_options[optimizer_type] for arg in optimizer_opts):
            raise ValueError("You tried to pass options incompatible with {} optimizer. "
                             "Check your parameters according to the description of the optimizer:\n\n{}".
                             format(optimizer_type, optimizers[optimizer_type].__doc__))

        optimizer = optimizers[optimizer_type](
            parameters,
            lr=hparams.learning_rate,
            weight_decay=hparams.weight_decay,
            **optimizer_opts
        )
    else:
        raise ValueError(f"`{optimizer_type}` is not a valid optimizer type")

    if hparams.with_lookahead:
        optimizer = torch_optimizer.Lookahead(optimizer, k=5, alpha=0.5)

    return optimizer


class FakeScheduler(optim.lr_scheduler._LRScheduler):
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", DeprecationWarning)

        return [group['lr'] for group in self.optimizer.param_groups]


class SchedulerTypes(str, Enum):
    none = "none"
    multi_step = "multi_step"
    exponential = "exp"
    plateau = "plateau"
    cyclic = "cyclic"


class ReduceLROnPlateau(optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):
        super(ReduceLROnPlateau, self).__init__(optimizer, mode, factor, patience,
                 verbose, threshold, threshold_mode, cooldown, min_lr, eps)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


    def get_last_lr(self):
        return self._last_lr


schedulers = {
    SchedulerTypes.none: FakeScheduler,
    SchedulerTypes.multi_step: optim.lr_scheduler.MultiStepLR,
    SchedulerTypes.exponential: optim.lr_scheduler.ExponentialLR,
    SchedulerTypes.plateau: ReduceLROnPlateau,
    SchedulerTypes.cyclic: optim.lr_scheduler.CyclicLR
}

schedulers_options = {
    SchedulerTypes.none: [],
    SchedulerTypes.multi_step: ["milestones", "gamma", "last_epoch"],
    SchedulerTypes.exponential: ["gamma", "last_epoch"],
    SchedulerTypes.plateau: ["mode", "factor", "patience", "threshold", "threshold_mode", "cooldown", "min_lr", "eps"],
    SchedulerTypes.cyclic: ["base_lr", "max_lr", "step_size_up", "step_size_down", "mode", "gamma", "scale_fn",
                            "scale_mode", "cycle_momentum", "base_momentum", "max_momentum", "last_epoch"]
}


def build_scheduler(optimizer, hparams):
    scheduler_type = SchedulerTypes[hparams.lr_scheduler]
    scheduler_opts = {} if hparams.lr_scheduler_options is None else hparams.lr_scheduler_options

    if scheduler_type in SchedulerTypes:
        if not all(arg in schedulers_options[scheduler_type] for arg in scheduler_opts):
            raise ValueError("You tried to pass options incompatible with {} lr scheduler. "
                             "Check your parameters according to the description of the scheduler:\n\n{}".
                             format(scheduler_type, schedulers[scheduler_type].__doc__))

        scheduler = schedulers[scheduler_type](
            optimizer,
            **scheduler_opts
        )
    else:
        raise ValueError(f"`{scheduler_type}` is not a valid optimizer type")

    return scheduler