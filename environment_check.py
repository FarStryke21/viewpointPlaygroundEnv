import gymnasium as gym
import numpy as np
import torch
import open3d as o3d
from gymnasium import spaces
import copy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.envs.registration import register
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
import sys
import os
import cv2
from datetime import datetime
sys.path.append("/home/dir/RL_CoveragePlanning/viewpointPlaygroundEnv/viewpoint_env")
from viewpointWorld import CoverageEnv

register(
    id="CoverageEnv-v0",
    entry_point="viewpoint_env.viewpointWorld:CoverageEnv",
)

env = CoverageEnv()

def visualize_viewpoint(mesh, observation, history):
    # Convert legacy TriangleMesh to tensor-based TriangleMesh
    t_mesh = mesh
    
    # Create a color array
    colors = np.full((len(t_mesh.triangle.indices), 3), [1, 1, 0])  # Yellow for unseen faces
    colors[history >= 1] = [0, 1, 0]
    colors[observation == 1] = [1, 0, 0]  # Red for seen faces
    
    # Assign colors to the mesh
    t_mesh.triangle.colors = o3d.core.Tensor(colors)
    
    o3d.visualization.draw([t_mesh])

obs, _ = env.reset()  # Reset the environment
done = False
total_reward = 0
mesh = env.mesh
i = 0
while not done:
    # Select a random action
    action = env.action_space.sample()
    
    # Step the environment with the selected action
    obs, reward, done, _, _ = env.step(action)
    
    # Accumulate total reward
    total_reward += reward
    
    visualize_viewpoint(mesh, obs, env.observation_history-obs)
    print(f"Iteration: {i:3} | Reward; {reward:10.2f} | Total Reward: {total_reward:10.2f}")
    i = i + 1
    if i == 10: break
    # Render the environment
    # env.render()

# Close the environment after testing
env.close()