import datetime as dt
import logging
import os

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

from easyRL import get_project_root
from easyRL.curstom_wrappers.callbacks import MakeGifCallback
from easyRL.prepare_env.prepare_env import prepare_env
from easyRL.utils.utils import *

log = logging.getLogger(__name__)


@hydra.main(
    config_path=os.path.join(get_project_root(), "config"),
    config_name="config.yaml",
    version_base=None,
)
def main(cfg: DictConfig):
    log.info("Start experiment!")

    log.info("Create folder to save all output e.g. model weights")
    algo_name = cfg.model._target_.split(".")[-1]
    env_name = cfg.env.env_name
    dirs = create_experiment_folder(
        path_to_outputs=cfg.path_to_outputs,
        algo_name=algo_name,
        env_name=env_name,
        description=cfg.description,
        experiment_name=cfg.experiment_name,
    )

    log_configs(path=dirs["params"], cfg=cfg)

    log.info(f"Start preparing training envs (numer of envs: {cfg.n_envs:_})")

    def make_custom_env():
        # TODO: fix it
        env = prepare_env(**cfg.env)
        return env

    env = make_vec_env(make_custom_env, n_envs=cfg.n_envs)
    env = VecMonitor(env, filename=dirs["logger"])

    log.info("Init MakeGifCallback and val env")
    val_env = make_custom_env()
    gif_callback = MakeGifCallback(
        save_freq=cfg.total_timesteps // cfg.eval_frequency // cfg.n_envs,
        save_path=dirs["outs_during_training"],
        eval_env=val_env,
    )

    log.info("Init CheckpointCallback")
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.total_timesteps // cfg.eval_frequency // cfg.n_envs,
        save_path=dirs["outs_during_training"],
        name_prefix="rl_model",
        verbose=2,
    )

    log.info(f"Preparing model (algo_name: {algo_name})")
    model = hydra.utils.instantiate(
        cfg.model,
        env=env,
        policy_kwargs=cfg.policy_net,
        tensorboard_log=dirs["tensorboard"],
    )

    actor_network = model.policy  # Actor network
    log.info(f"Policy architecture: {actor_network}")

    log.info("##################### START TRAINING #####################")
    message = f"Training starts on: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    log_to_description(message=message, path=dirs["experiment_path"])

    model.learn(
        total_timesteps=cfg.total_timesteps,
        log_interval=None,
        callback=[checkpoint_callback, gif_callback],
    )

    log.info("Training done!")
    message = f"Training ends on: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    log_to_description(path=dirs["experiment_path"], message=message)
    log_training_time(path=dirs["experiment_path"])

    log.info("Start evaluating model")
    evaluate_and_make_gif(val_env, model, cfg.eval_games_number, dirs)

    log.info("Creating final plots")
    create_plots()  # TODO: create plots

    log.info("Everything is done! Bye! bye!")


if __name__ == "__main__":
    main()
