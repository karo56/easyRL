# Here is place to create and add your custom env
# See some inspiration here: https://www.gymlibrary.dev/content/environment_creation/

import gymnasium as gym


class MyCustomEnv(gym.Env):
    def __init__(self, render_mode=None, size=5):
        pass

    def _get_obs(self):
        pass

    def _get_info(self):
        pass

    def reset(self, seed=None, options=None):
        pass

    def step(self, action):
        pass

    def render(self):
        pass

    def _render_frame(self):
        pass

    def close(self):
        pass
