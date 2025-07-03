import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from src import utils
from src.utils.lr_scheduler import get_scheduler
from src.utils.optim import get_optimizer
from pytorch_lightning import LightningModule
from torch import distributed as dist
from torch import nn
from torchmetrics import MaxMetric, MeanMetric, Metric, MinMetric, SumMetric

log = utils.get_logger(__name__)


@contextmanager
def on_prediction_mode(pl_module: LightningModule, enable=True):
    if not enable:
        yield
        return

    _methods = [
        '{}_step',
    ]

    for _method in _methods:
        _test_method, _predict_method = _method.format('test'), _method.format('predict')

        _test_method_obj = getattr(pl_module, _test_method, None)
        _predict_method_obj = getattr(pl_module, _predict_method, None)

        setattr(pl_module, _test_method, _predict_method_obj)
        setattr(pl_module, _predict_method, _test_method_obj)

    yield

    for _method in _methods:
        _test_method, _predict_method = _method.format('test'), _method.format('predict')

        _test_method_obj = getattr(pl_module, _test_method, None)
        _predict_method_obj = getattr(pl_module, _predict_method, None)

        setattr(pl_module, _test_method, _predict_method_obj)
        setattr(pl_module, _predict_method, _test_method_obj)


class TaskLitModule(LightningModule):
    def __init__(
        self,
        model: List[nn.Module],
        criterion: nn.Module = None,
        optimizer: Union[Callable, torch.optim.Optimizer] = None,
        lr_scheduler: Union[Callable, torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=True)

        self.criterion = criterion
        self.valid_logged = {}

    def setup(self, stage=None) -> None:
        self._stage = stage
        super().setup(stage)

    @property
    def lrate(self):
        for param_group in self.trainer.optimizers[0].param_groups:
            return param_group['lr']

    @property
    def stage(self):
        return self._stage

    def log(self, name: str, value, prog_bar: bool = False, logger: bool = True,
            on_step: Optional[bool] = None, on_epoch: Optional[bool] = None, **kwargs) -> None:
        if self.trainer and not self.trainer.training and on_epoch:
            self.valid_logged[name] = value
        return super().log(name, value, prog_bar=prog_bar, logger=logger,
                           on_step=on_step, on_epoch=on_epoch, **kwargs)

    # -------# Training #-------- #
    def step(self, batch):
        raise NotImplementedError

    def training_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError

    def on_train_epoch_end(self) -> None:
        # Epoch-related logging or metrics reset can go here
        if dist.is_initialized() and hasattr(self.trainer.datamodule, 'train_batch_sampler'):
            self.trainer.datamodule.train_batch_sampler.set_epoch(self.current_epoch + 1)
            self.trainer.datamodule.train_batch_sampler._build_batches()

    # -------# Evaluating #-------- #
    def validation_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError

    def on_validation_epoch_end(self) -> None:
        logging_info = ", ".join(f"{key}={val:.3f}" for key, val in self.valid_logged.items())
        log.info(f"Validation Info @ (Epoch {self.current_epoch}, global step {self.global_step}): {logging_info}")

    def test_step(self, batch: Any, batch_idx: int):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end()

    # -------# Inference/Prediction #-------- #
    def forward(self, batch):
        raise NotImplementedError

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        raise NotImplementedError

    def on_predict_epoch_end(self, results: List[Any], log_pref=None) -> None:
        raise NotImplementedError

    # -------# Optimizers & Lr Schedulers #-------- #
    def configure_optimizers(self):
        optimizer = get_optimizer(self.hparams.optimizer, self.parameters())
        if 'lr_scheduler' in self.hparams and self.hparams.lr_scheduler is not None:
            lr_scheduler, extra_kwargs = get_scheduler(self.hparams.lr_scheduler, optimizer)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {"scheduler": lr_scheduler, **extra_kwargs}
            }
        return optimizer


class AutoMetric(nn.Module):
    _type_shortnames = dict(
        mean=MeanMetric,
        sum=SumMetric,
        max=MaxMetric,
        min=MinMetric,
    )

    def __init__(self) -> None:
        super().__init__()
        self.register_parameter('_device', torch.zeros(1))

    @property
    def device(self):
        return self._device.device

    def update(self, name, value, type='mean', **kwds):
        if not hasattr(self, name):
            if isinstance(type, str):
                type = self._type_shortnames[type]
            setattr(self, name, type(**kwds))
            getattr(self, name).to(self.device)

        getattr(self, name).update(value)

    def compute(self, name):
        return getattr(self, name).compute()

    def reset(self, name):
        getattr(self, name).reset()


TASK_REGISTRY = {}

def register_task(name):
    def decorator(cls):
        cls._name_ = name
        TASK_REGISTRY[name] = cls
        return cls
    return decorator


# Automatically import all task modules
utils.import_modules(os.path.dirname(__file__), "src.tasks")