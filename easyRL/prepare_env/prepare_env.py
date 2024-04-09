import logging
from typing import List

import gymnasium as gym
from gymnasium import ActionWrapper, ObservationWrapper, RewardWrapper

log = logging.getLogger(__name__)


def prepare_env(
    env_name: str,
    render_mode: str = "rgb_array",
    list_of_observation_wrappers: List[ObservationWrapper] = None,
    list_of_action_wrappers: List[ActionWrapper] = None,
    list_of_reward_wrappers: List[: List[RewardWrapper]] = None,
) -> gym.Env:
    if list_of_observation_wrappers is None:
        list_of_observation_wrappers = []
    if list_of_action_wrappers is None:
        list_of_action_wrappers = []
    if list_of_reward_wrappers is None:
        list_of_reward_wrappers = []

    env = gym.make(env_name, render_mode=render_mode)

    for obs_wrapper in list_of_observation_wrappers:
        env = obs_wrapper(env)

    for act_wrapper in list_of_action_wrappers:
        env = act_wrapper(env)

    for rew_wrapper in list_of_reward_wrappers:
        env = rew_wrapper(env)

    return env
