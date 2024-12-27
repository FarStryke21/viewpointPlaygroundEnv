import os
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

import torch
import open3d as o3d

from numba import jit

@jit(nopython=True)
def process_observation(primitive_ids, num_triangles, invalid_id):
    observation = np.zeros(num_triangles)
    for id in primitive_ids:
        if id != invalid_id:
            observation[id] = 1
    return observation

@jit(nopython=True)
def jit_calculate_intrinsic_reward(last_observation, observation_history, num_triangles):
    observation_change = np.sum(np.logical_and(last_observation, np.logical_not(observation_history)))
    normalized_change = observation_change / num_triangles
    total_observed = np.sum(observation_history)
    novelty = 1 / (np.sqrt(total_observed) + 1)
    return (normalized_change + novelty) / 2

@jit(nopython=True)
def jit_get_pose(action, sensor_range):
    theta, phi = action[0], action[1]
    x = sensor_range * np.sin(phi) * np.cos(theta)
    y = sensor_range * np.sin(phi) * np.sin(theta)
    z = sensor_range * np.cos(phi)
    return np.array([x, y, z])

@jit(nopython=True)
def jit_get_reward(newly_covered, coverage, done_val, action_history_length, intrinsic_reward, reward_weights):
    r = np.array([
        np.sum(newly_covered),
        1 if coverage >= done_val else 0,
        action_history_length,
        intrinsic_reward
    ])
    return np.dot(reward_weights, r)

class CoverageEnv(gym.Env):
    """
    A custom Gymnasium environment for 3D mesh coverage planning.
    
    This environment simulates an agent tasked with observing as much of a 3D mesh
    as possible from different viewpoints.
    """

    def __init__(self, mesh_folder='/home/dir/RL_CoveragePlanning/test_models/modified',
                  sensor_range=25, fov_deg=60, width_px=640, height_px=480, 
                  coverage_req=0.90,
                  render_mode='rgb_array', 
                  train = True,
                  save_action_history=True, 
                  save_path = '/home/dir/RL_CoveragePlanning/action'
                ):
        """
        Initialize the CoverageEnv.

        Args:
            mesh_folder (str): Path to the folder containing mesh files.
            sensor_range (float): Range of the sensor.
            fov_deg (float): Field of view in degrees.
            width_px, height_px (int): Width and height of the rendered image in pixels.
            coverage_req (float): Required coverage to consider the task complete.
            render_mode (str): Rendering mode for the environment.
            train (bool): Whether the environment is being used for training.
            save_action_history (bool): Whether to save the history of actions.
            save_path (str): Path to save the action history.
        """

        super(CoverageEnv, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        
        self.save_action_history = save_action_history
        self.save_path = save_path

        # Load mesh file
        self.mesh_file_name = self._get_mesh_file(mesh_folder) if train else 'test_8.obj'
        self.mesh_file = os.path.join(mesh_folder, self.mesh_file_name)
        print(f"Mesh file: {self.mesh_file_name} loaded for environment...")

        # Set up the mesh and scene
        self.mesh = o3d.io.read_triangle_mesh(self.mesh_file)
        self.mesh.translate(-self.mesh.get_center())
        self.mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(self.mesh)

         # Set up environment parameters
        self.fov_deg, self.width_px, self.height_px = fov_deg, width_px, height_px
        self.up, self.center = [0, -1, 0], [0, 0, 0]
        self.agent_pose = np.zeros(3)
        self.done_val = coverage_req
        self.sensor_range = sensor_range
        self.bbox = self.mesh.get_axis_aligned_bounding_box()
        self.INVALID_ID = 4294967295
        self.num_triangles = self.mesh.triangle.indices.shape[0]
        self.tracker = None

        # Set up the action space and observation space
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict({
                    'last_observation': spaces.Box(low=0, high=1, shape=(self.num_triangles,), dtype=np.float32),
                    'observation_history': spaces.Box(low=0, high=1, shape=(self.num_triangles,), dtype=np.float32),
                    'coverage_percentage': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                    'last_viewpoint': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
                    })
        self.last_observation = np.zeros((self.num_triangles,), dtype=np.float32)
        self.observation_history = np.zeros((self.num_triangles), dtype=np.float32)
        self.percentage_covered = 0.0
        self.last_viewpoint = np.zeros(3, dtype=np.float32)
        self.action_history = []
        
        # Set up the reward weights
        self.reward_weights = np.array([
            100,                    # w_newly_covered
            10.0,                   # w_threshold_bonus
            -1.0,                   # w_step_penalty
            10                      # w_intrinsic_reward
        ])

        

    def reset(self, seed=None, options=None):
        """
        Reset the environment.

        Args:
            seed (int): Seed for the environment.
            options (dict): Options for the environment.

        Returns:
            dict: Initial observation.
        """

        # Randomize the initial agent pose
        super().reset(seed=seed)

        # print("Resetting the environment...")
        self.agent_pose = self.action_space.sample()
        self.last_observation = np.zeros((self.num_triangles,), dtype=np.float32)
        self.observation_history = np.zeros((self.num_triangles,), dtype=np.float32)
        self.percentage_covered = 0.0
        self.last_viewpoint = np.zeros(3, dtype=np.float32)
        self.action_history = []

        return self._get_obs(), {}

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action (np.ndarray): Action to take.

        Returns:
            dict: Observation after the step.
            float: Reward after the step.
            bool: Whether the task is terminated.
            bool: Whether the episode is truncated.
            dict: Additional information.
        """

        action = self._get_pose(action)    
        self.action_history.append(action)
        self.agent_pose = action

        self.tracker = self._get_observation(self.agent_pose)
        self.last_observation = self._process_observation(self.tracker) 
        newly_covered = np.logical_and(self.last_observation, np.logical_not(self.observation_history))
        self.observation_history = np.maximum(self.observation_history, self.last_observation)
        self.percentage_covered = np.mean(self.observation_history)

        reward = self.get_reward(newly_covered, self.percentage_covered)
        terminated = self.percentage_covered >= self.done_val
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}
    
    def _get_obs(self):
        """
        Get the current observation.

        Returns:
            dict: Current observation.
        """
        return {
            'last_observation': self.last_observation,
            'observation_history': self.observation_history,
            'coverage_percentage': np.array([self.percentage_covered]),
            'last_viewpoint': self.last_viewpoint
        }

    def _get_mesh_file(self, folder):
        """
        Get the mesh file from the folder.

        Args:
            folder (str): Folder containing the mesh files.

        Returns:
            str: Mesh file name.
        """
        # Get the file path of the mesh file
        files = os.listdir(folder)
        return np.random.choice(files)

    def _get_pose(self, action):
        """
        Get the pose from the action.

        Args:
            action (np.ndarray): Action to take.

        Returns:
            np.ndarray: Pose from the action.
        """
        return jit_get_pose(action, self.sensor_range)
    
    def _get_observation(self, pose):
        """
        Get the observation from the pose.

        Args:
            pose (np.ndarray): Pose to get the observation from.

        Returns:
            dict: Observation from the pose.
        """

        rays = self.scene.create_rays_pinhole(fov_deg=self.fov_deg,
                                center = self.center,
                                eye = pose,
                                up = self.up,
                                width_px=self.width_px,
                                height_px=self.height_px)
        result = self.scene.cast_rays(rays)
        return result 
    
    def _process_observation(self, tracker):
        """
        Process the observation from the tracker.

        Args:
            tracker (dict): Tracker containing the observation.

        Returns:
            np.ndarray: Processed observation.
        """
        primitive_ids = tracker['primitive_ids'].numpy().reshape(-1)
        return process_observation(primitive_ids, self.num_triangles, self.INVALID_ID)

    def _calculate_intrinsic_reward(self):
        """
        Calculate the intrinsic reward based on observation novelty and change.

        Returns:
            float: Intrinsic reward value.
        """
        return jit_calculate_intrinsic_reward(self.last_observation, self.observation_history, self.num_triangles)


    def get_reward(self, newly_covered, coverage):
        """
        Get the reward from the newly covered faces and coverage.

        Args:
            newly_covered (np.ndarray): Newly covered faces.
            coverage (float): Coverage percentage.

        Returns:
            float: Reward from the newly covered faces and coverage.
        """
        intrinsic_reward = self._calculate_intrinsic_reward()
        return jit_get_reward(newly_covered, coverage, self.done_val, len(self.action_history), intrinsic_reward, self.reward_weights)

    
    def render(self):
        """
        Render the environment.

        Returns:
            np.ndarray: Rendered image.
        """
        plt.imshow(self.tracker['t_hit'].numpy())

    def close(self):
        """
        Close the environment.

        Returns:
            None
        """

        if self.save_action_history:
            self.pose_path = os.path.join(self.save_path, f"{self.mesh_file_name.split('.')[0]}_poses.csv")
            print("Saving the action history...")
            np.savetxt(self.pose_path, np.unique(self.action_history, axis=0), delimiter=",")
            print("Percentage covered: ", (self.percentage_covered*100))
            print(f"Number of actions: {len(self.action_history)}")

