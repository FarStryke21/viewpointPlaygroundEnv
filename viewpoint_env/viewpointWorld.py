import gymnasium as gym
import numpy as np
import open3d as o3d
from gymnasium import spaces
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

class HollowCuboidActionSpace(gym.spaces.Box):
    def __init__(self, low, high, thickness):
        self.low = low
        self.high = high
        self.thickness = thickness
        self.outer_space = gym.spaces.Box(low=low - thickness, high=high + thickness)
        self.inner_space = gym.spaces.Box(low=low, high=high)
        super().__init__(low - thickness, high + thickness, dtype=np.float32)
        

    def sample(self):
        action = self.outer_space.sample()
        if not self.inner_space.contains(action):
            return action
        else:
            return self.sample()

    def contains(self, x):
        return self.outer_space.contains(x) and not self.inner_space.contains(x)

    def __repr__(self):
        return "HollowCuboidActionSpace({}, {}, thickness={})".format(self.low, self.high, self.thickness)

class CoverageEnv(gym.Env):
    def __init__(self, mesh_file='/home/aman/Desktop/RL_CoveragePlanning/viewpointPlaygroundEnv/meshes/stanford_bunny.obj',
                  sensor_range=0.1, fov_deg=60, width_px=320, height_px=240, coverage_req=0.99,
                  render_mode='rgb_array',
                  save_action_history=True, save_path = '/home/aman/Desktop/RL_CoveragePlanning/action/poses.csv'):
        super(CoverageEnv, self).__init__()

        self.save_action_history = save_action_history
        self.save_path = save_path

        self.mesh = o3d.io.read_triangle_mesh(mesh_file)
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
        low = self.bbox.min_bound.numpy()
        high = self.bbox.max_bound.numpy()

        self.action_space = HollowCuboidActionSpace(low, high, self.sensor_range)

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
        primitive_ids = primitive_ids[primitive_ids != self.INVALID_ID]
        primitive_ids = np.unique(primitive_ids)
        
        # Create a zero array of size num_triangles
        observation = np.zeros(self.num_triangles)
        # Set the elements in observation to 1 where the primitive_ids are present
        observation[primitive_ids] = 1
        return observation

    def get_reward(self):
        # Subtract observation history from the current observation space
        reward_list = self.observation_space - self.observation_history
        # Find number of positive elements in the reward list
        mask = reward_list > 0
        new_covered = mask.sum()
        percentage_new = new_covered / self.observation_space.shape[0]
        return (-100*len(self.action_history)) + (percentage_new*100)
    
    def render(self):
        plt.imshow(self.tracker['t_hit'].numpy())

    def close(self):
        if self.save_action_history:
            print("Saving the action history...")
            np.savetxt(self.save_path, self.action_history, delimiter=",")
        # print("Summary of the coverage")
        # print("-----------------------------------------------")
        # print(f"Percentage of mesh covered: {self.percentage_covered*100}%")
        # print(f"Number of steps: {len(self.action_history)}")
        # print("\n")
