#environment utils go here

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim

#create the atari wrapper

import gymnasium as gym
import numpy as np
from collections import deque

class AtariEnvWrapper:
    def __init__(self, env_name="Breakout-v5", render_mode="rgb_array", stack_size=4):
        self.env = gym.make(env_name, render_mode=render_mode)
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)

    def preprocess(self, frame):
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 110), interpolation=cv2.INTER_AREA)
        cropped = resized[18:102, :]
        return cropped.astype(np.uint8)  # (84, 84)

    def reset(self):
        obs, _ = self.env.reset()
        frame = self.preprocess(obs)
        self.frames = deque([frame] * self.stack_size, maxlen=self.stack_size)
        return np.stack(self.frames, axis=0)  # (4, 84, 84)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = self.preprocess(obs)
        self.frames.append(frame)
        stacked_obs = np.stack(self.frames, axis=0)  # (4, 84, 84)
        done = terminated or truncated
        return stacked_obs, reward, done, info

env = AtariEnvWrapper("ALE/Breakout-v5")
obs = env.reset()
# print(obs.shape) #(4, 84, 84) stack of 4 frames

action = env.env.action_space.sample()   # env.env to access underlying Gym env
obs, reward, done, info = env.step(action)

#summon the env
env = AtariEnvWrapper("ALE/Breakout-v5")
obs = env.reset()
