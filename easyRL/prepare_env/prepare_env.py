import logging

import gymnasium as gym

log = logging.getLogger(__name__)


def prepare_env(
    env_name,
    render_mode="rgb_array",
    list_of_observation_wrappers=[],
    list_of_action_wrappers=[],
    list_of_reward_wrappers=[],
):  # TODO: add hint typing and fix lists
    env = gym.make(env_name, render_mode=render_mode)

    for obs_wrapper in list_of_observation_wrappers:
        env = obs_wrapper(env)

    for act_wrapper in list_of_action_wrappers:
        env = act_wrapper(env)

    for rew_wrapper in list_of_reward_wrappers:
        env = rew_wrapper(env)

    return env
