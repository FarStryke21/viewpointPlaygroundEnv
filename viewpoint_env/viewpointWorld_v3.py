import gymnasium as gym
import numpy as np
import torch
import open3d as o3d
from gymnasium import spaces
import os
import matplotlib.pyplot as plt

class CoverageEnv(gym.Env):
    def __init__(self, mesh_folder='/home/dir/RL_CoveragePlanning/test_models/modified',
                  sensor_range=50, fov_deg=60, width_px=640, height_px=480, 
                  coverage_req=0.90,
                  render_mode='rgb_array', 
                  train = False,
                  save_action_history=True, 
                  save_path = '/home/dir/RL_CoveragePlanning/action'
                ):
        
        super(CoverageEnv, self).__init__()

        self.save_action_history = save_action_history
        self.save_path = save_path

        # if model is being trained, select a random mesh file from the folder
        if train:
            self.mesh_file_name = self.get_mesh_file(mesh_folder)
            self.mesh_file = os.path.join(mesh_folder, self.mesh_file_name)
        else:
            self.mesh_file_name = 'test_6.obj'
            self.mesh_file = os.path.join(mesh_folder, self.mesh_file_name)

        print(f"Mesh file: {self.mesh_file_name} loaded for environment...")

        self.mesh = o3d.io.read_triangle_mesh(self.mesh_file)
        # self.mesh.scale(0.01, center=self.mesh.get_center())
        self.mesh.translate(-self.mesh.get_center())
        
        self.mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)

        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(self.mesh)

        self.fov_deg = fov_deg
        self.width_px = width_px
        self.height_px = height_px
        self.up = [0, -1, 0]
        self.center = [0, 0, 0]

        self.agent_pose = np.array([0.0, 0.0, 0.0])  # [x, y, z]
        self.done_val = coverage_req

        self.sensor_range = sensor_range
        self.bbox = self.mesh.get_axis_aligned_bounding_box()

        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(2,), dtype=np.float32)

        self.INVALID_ID = 4294967295

        self.num_triangles = self.mesh.triangle.indices.shape[0]
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_triangles,), dtype=np.float32)
        self.observation_history = np.zeros((self.num_triangles), dtype=np.float32)
        self.percentage_covered = 0.0
        self.tracker = None

        self.action_history = []

    def reset(self, seed=None, options=None):
        # Randomize the initial agent pose
        super().reset(seed=seed)
        # print("Resetting the environment...")
        self.agent_pose = self.action_space.sample()
        self.observation_history = np.zeros((self.num_triangles), dtype=np.float32)
        self.percentage_covered = 0.0
        self.tracker = None
        self.action_history = []

        return self.observation_history, {}

    def step(self, action):

        action = self.get_pose(action)    
        # print(f"Action: {action} | Valid: {self.action_space.contains(action)}")
        self.action_history.append(action)
        self.agent_pose = action
        # print(f"Agent pose: {self.agent_pose}")
        self.tracker = self.get_observation(self.agent_pose)
        self.observation_space = self.process_observation(self.tracker) 
        self.observation_history = self.observation_space + self.observation_history
        
        # Calculate the percentage of the mesh covered by the percent of non zero elements in the observation space
        self.percentage_covered = np.count_nonzero(self.observation_history) / self.observation_history.shape[0]
        reward = self.get_reward()
        terminated = self.percentage_covered >= self.done_val
        truncated = False

        # Step returns observation of state, reward, done, and info in a tuple
        # print(f"Count {len(self.action_history)}")
        # print(f"Reward: {reward}")
        # print(f"Covered in current step: {(np.count_nonzero(self.observation_space) / self.observation_space.shape[0])*100}%")
        # print(f"Total Percentage covered: {self.percentage_covered*100}% \n")
        
        return self.observation_space, reward, terminated, truncated, {}
    
    def get_mesh_file(self, folder):
        # Get the file path of the mesh file
        files = os.listdir(folder)
        # randomly select a file from the folder
        return np.random.choice(files)

    def get_pose(self, action):
        # Modify action here
        theta, phi = action[0], action[1]
        x = self.sensor_range * np.sin(phi) * np.cos(theta)
        y = self.sensor_range * np.sin(phi) * np.sin(theta)
        z = self.sensor_range * np.cos(phi)

        modified_action = np.array([x, y, z])
        return modified_action
    
    def get_observation(self, pose):
        rays = self.scene.create_rays_pinhole(fov_deg=self.fov_deg,
                                center = self.center,
                                eye = pose,
                                up = self.up,
                                width_px=self.width_px,
                                height_px=self.height_px)
        result = self.scene.cast_rays(rays)
        return result 
    
    def process_observation(self, tracker):
        primitive_ids = tracker['primitive_ids'].numpy()

        # Reshape primitive_ids to a 1D array
        primitive_ids = primitive_ids.reshape(-1)

        # Remove all the invalid IDs
        primitive_ids = np.unique(primitive_ids[primitive_ids != self.INVALID_ID])
        
        # Create a zero array of size num_triangles
        observation = np.zeros(self.num_triangles)
        # Set the elements in observation to 1 where the primitive_ids are present
        observation[primitive_ids] = 1
        
        return observation

    def get_reward(self):
        # Subtract observation history from the current observation space
        past_history = self.observation_history-self.observation_space
        xor_result = np.logical_xor(past_history, self.observation_history)
        new_covered = (np.sum(xor_result) / self.observation_space.shape[0])*100
        # get the mean position of all actions
        mean_action = np.mean(self.action_history, axis=0)
        # find disance of mean from the center of the mesh
        distance = np.linalg.norm(mean_action)
        reward = (-10*len(self.action_history)) + (new_covered*100) + ((1/distance)*100)
        reward = (new_covered*10) 
        # print(f"Reward: {reward}")
        return reward
    
    def render(self):
        plt.imshow(self.tracker['t_hit'].numpy())

    def close(self):
        if self.save_action_history:
            self.pose_path = os.path.join(self.save_path, f"{self.mesh_file_name.split('.')[0]}_poses.csv")
            print("Saving the action history...")
            np.savetxt(self.pose_path, np.unique(self.action_history, axis=0), delimiter=",")
            print("Percentage covered: ", (self.percentage_covered*100))
            print(f"Number of actions: {len(self.action_history)}")