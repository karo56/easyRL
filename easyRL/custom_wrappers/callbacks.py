import glob
import logging
import os

import imageio
import pandas as pd
from gymnasium import Env
from stable_baselines3.common.callbacks import BaseCallback

log = logging.getLogger(__name__)


class MakeGifCallback(BaseCallback):
    def __init__(
        self,
        eval_env: Env,
        save_path: str,
        save_freq: str,
    ):
        super(MakeGifCallback, self).__init__()
        self.eval_env = eval_env
        self.save_path = save_path

        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self._make_gif()

        return True

    def _make_gif(self) -> None:
        stats = {
            "episodes_rews": [],
            "episodes_lens": [],
        }
        done = False
        obs, _info = self.eval_env.reset()
        # note: obs is tuple: (obs, info), so we should take only obs
        # see here: https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html  # noqa

        images = []
        images.append(self.eval_env.render())
        ep_rew, ep_len = 0, 0

        while not done:
            action, _states = self.model.predict(obs)

            obs, rewards, done, truncated, _info = self.eval_env.step(action)

            # 'done' don't change when max_episode_steps riches :(
            done = done or truncated

            images.append(self.eval_env.render())
            ep_rew += rewards
            ep_len += 1

        stats["episodes_lens"].append(ep_len)
        stats["episodes_rews"].append(ep_rew)

        n_files = len(glob.glob(os.path.join(self.save_path, "*.gif")))
        stats_df = pd.DataFrame(stats)
        stats_df.to_csv(
            os.path.join(
                self.save_path, f"stats_on_step_{self.n_calls}_{n_files+1}.csv"
            )
        )
        imageio.mimsave(
            os.path.join(
                self.save_path, f"gif_game_on_step_{self.n_calls}_{n_files+1}.gif"
            ),
            images,
        )
        log.info(f"stats on step {self.n_calls}: {stats}")
