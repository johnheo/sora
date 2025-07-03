import os
from typing import List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import (
    Trainer,
    seed_everything,
)

import pyrootutils

from src import utils

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` is recommended at the top of each start file
# to make the environment more robust and consistent
#
# the line above searches for ".git" or "pyproject.toml" in present and parent dirs
# to determine the project root dir
#
# adds root dir to the PYTHONPATH (if `pythonpath=True`)
# so this file can be run from any place without installing project as a package
#
# sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
# this makes all paths relative to the project root
#
# additionally loads environment variables from ".env" file (if `dotenv=True`)
#
# you can get away without using `pyrootutils.setup_root(...)` if you:
# 1. move this file to the project root dir or install project as a package
# 2. modify paths in "configs/paths/default.yaml" to not use PROJECT_ROOT
# 3. always run this file from the project root dir
#
# https://github.com/ashleve/pyrootutils
# -------------------

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    # load environment variables from `.env` file if it exists
    # recursively searches for `.env` in all folders starting from work dir
    dotenv=True,
)

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains the training pipeline. Can additionally evaluate model on a testset, using best
    weights achieved during training.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Convert relative ckpt path to absolute path if necessary
    ckpt_path = not config.train.get("force_restart", False) and config.train.get("ckpt_path")
    if ckpt_path:
        ckpt_path = utils.resolve_ckpt_path(ckpt_dir=config.paths.ckpt_dir, ckpt_path=ckpt_path)
        if os.path.exists(ckpt_path):
            log.info(f"Resuming checkpoint from <{ckpt_path}>")
        else:
            log.info(f"Failed to resume checkpoint from <{ckpt_path}>: file not exists. Skip.")
            ckpt_path = None

    # loading pipeline
    OmegaConf.set_struct(config, False)  # disables struct mode recursively
    datamodule, pl_module, logger, callbacks = utils.common_pipeline(config)

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    # trainer: Trainer = hydra.utils.instantiate(
    #     config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    # )
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=[], logger=logger, _convert_="partial", enable_model_summary=False
    )
    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        datamodule=datamodule,
        # model=model,
        model=pl_module,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    if config.get("train"):
        log.info("Starting training!")
        trainer.fit(model=pl_module, datamodule=datamodule, ckpt_path=ckpt_path)

    # Get metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric and optimized_metric not in trainer.callback_metrics:
        raise Exception(
            "Metric for hyperparameter optimization not found! "
            "Make sure the `optimized_metric` in `hparams_search` config is correct!"
        )
    score = trainer.callback_metrics.get(optimized_metric)

    # Test the model
    if config.get("test"):
        log.info("Starting testing!")
        best_ckpt_path = os.path.join(config.paths.ckpt_dir, 'best.ckpt')
        trainer.test(model=pl_module, datamodule=datamodule, ckpt_path=best_ckpt_path)

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=pl_module,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run") and config.get("train"):
        log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    return score


@hydra.main(version_base='1.1', config_path=f"{root}/configs", config_name="train.yaml")
def main(config: DictConfig):
    # Pretty print the config
    print("=" * 80)
    print(OmegaConf.to_yaml(config, resolve=True))
    print("=" * 80)

    return train(config)


if __name__ == "__main__":
    main()

# python train.py hydra/job_logging=none hydra/hydra_logging=none trainer.logger=false