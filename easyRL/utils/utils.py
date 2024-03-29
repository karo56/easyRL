import datetime as dt
import logging
import os

import imageio
import pandas as pd


log = logging.getLogger(__name__)

def create_experiment_folder(
    path_to_outputs,
    algo_name,
    description="This is base description of experiment.",
    experiment_name="name",
):
    final_path = os.path.join(path_to_outputs, algo_name)

    if not os.path.isdir(path_to_outputs):
        os.mkdir(path_to_outputs)

    if not os.path.isdir(final_path):
        os.mkdir(final_path)

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

def log_to_description(path, message):
    log.info(message)
    text_file = open(os.path.join(path,"description.txt"), "a")
    text_file.write("\n"+message)
    text_file.close()


def evaluate_and_make_gif(env, model, n_games, path):
    stats = {
        "episodes_rews": [],
        "episodes_lens": [],
    }

    for game in range(1, n_games+1):
        log.info(f"Game nr {game}")

        done = False
        obs, _info = env.reset()

        images = []
        images.append(env.render())
        ep_rew, ep_len = 0, 0

        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, truncated, _info= env.step(action)

            # 'done' don't change when max_episode_steps riches :(
            done = done or truncated

            images.append(env.render())
            ep_rew += rewards
            ep_len += 1

        stats["episodes_lens"].append(ep_len)
        stats["episodes_rews"].append(ep_rew)

        imageio.mimsave(
            os.path.join(path, f"gif_game_nr_{game}.gif"), images
        )

    log.info(f" TODO: write something about it {stats}")
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(os.path.join(path, "stats_eval.csv"))

    return stats