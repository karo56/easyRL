import gymnasium as gym
import numpy as np

class NormalizeObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        return observation / self.env.observation_space.high

class ClipActionWrapper(gym.ActionWrapper):
    def action(self, action):
        return np.clip(action, self.action_space.low, self.action_space.high)


class CustomRewardMountainCar(gym.RewardWrapper):
    def reward(self, reward):
        reward += 13 * abs(self.env.state[1])
        return reward