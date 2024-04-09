import datetime as dt
import logging
import os

import imageio
import pandas as pd
from gymnasium import Env
from omegaconf import OmegaConf
from stable_baselines3.common.base_class import BaseAlgorithm

log = logging.getLogger(__name__)


def create_experiment_folder(
    path_to_outputs: str,
    algo_name: str,
    env_name: Env,
    description: str = "This is base description of experiment.",
    experiment_name: str = "name",
) -> dict:
    final_path = os.path.join(path_to_outputs, env_name, algo_name)
    log.info(f"final_path: {final_path}")

    if not os.path.exists(final_path):
        os.makedirs(final_path)

    list_of_experiments = [f.path for f in os.scandir(final_path) if f.is_dir()]
    list_of_experiments.sort()

    if len(list_of_experiments) == 0:
        experiment_id = 0
    else:
        experiment_id = os.path.basename(list_of_experiments[-1]).split("_")[0]

    new_id = str(int(experiment_id) + 1).zfill(3) + "_" + experiment_name

    new_path = os.path.join(final_path, new_id)
    os.mkdir(new_path)

    list_of_folders = [
        "plots",
        "logger",
        "tensorboard",
        "params",
        "model",
        "gifs",
        "outs_during_training",
    ]
    dirs = {}
    dirs["experiment_path"] = new_path

    for folder in list_of_folders:
        path = os.path.join(new_path, folder)
        dirs[folder] = path

        if folder == "model":
            continue
        os.mkdir(path)

    text_file = open(os.path.join(new_path, "description.txt"), "w")
    text_file.write(description)
    text_file.write(
        f"\nExperiment performed on {dt.datetime.now().strftime('%b %d %Y %H:%M:%S')}"
    )
    text_file.close()

    return dirs


def log_configs(path: str, cfg: dict) -> None:
    log.info(f"Experiment starts with config: {cfg}")
    output_path = os.path.join(path, "config.yaml")
    with open(output_path, "w") as f:
        OmegaConf.save(cfg, f)


def log_to_description(path: str, message: str) -> None:
    log.info(message)
    text_file = open(os.path.join(path, "description.txt"), "a")
    text_file.write("\n" + message)
    text_file.close()


def log_training_time(path: str) -> None:
    text_file = open(os.path.join(path, "description.txt"), "r+")
    lines = text_file.readlines()

    start_time = dt.datetime.strptime(
        lines[2].strip(), "Training starts on: %Y-%m-%d %H:%M:%S"
    )
    end_time = dt.datetime.strptime(
        lines[3].strip(), "Training ends on: %Y-%m-%d %H:%M:%S"
    )

    training_time = end_time - start_time
    training_minutes = training_time.total_seconds() / 60

    message = f"\nTraining time: {int(training_minutes)} minutes"
    log.info(message)

    text_file.write(message)
    text_file.close()


def evaluate_and_make_gif(
    env: Env, model: BaseAlgorithm, n_games: int, dirs: dict
) -> None:
    path_gif = dirs["gifs"]
    path_logger = dirs["logger"]
    stats = {
        "episodes_rews": [],
        "episodes_lens": [],
    }

    for game in range(1, n_games + 1):
        log.info(f"Game nr {game}")

        done = False
        obs, _info = env.reset()

        images = []
        images.append(env.render())
        ep_rew, ep_len = 0, 0

        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, truncated, _info = env.step(action)

            # 'done' don't change when max_episode_steps riches :(
            done = done or truncated

            images.append(env.render())
            ep_rew += rewards
            ep_len += 1

        stats["episodes_lens"].append(ep_len)
        stats["episodes_rews"].append(ep_rew)

        imageio.mimsave(os.path.join(path_gif, f"gif_game_nr_{game}.gif"), images)

    log.info(f" TODO: write something about it {stats}")
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(os.path.join(path_logger, "df_logger_val.csv"))


def create_plots() -> None:
    # TODO: add plots
    pass
