import gymnasium as gym
import numpy as np
import trimesh
from gymnasium import spaces
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

class CoverageEnv(gym.Env):
    def __init__(self, mesh_file, sensor_range, sensor_resolution):
        super(CoverageEnv, self).__init__()

        self.mesh = trimesh.load(mesh_file)
        self.sensor_range = sensor_range
        self.sensor_resolution = sensor_resolution
        self.agent_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # [x, y, z, alpha, beta, gamma]

        self.mesh_bbox = self.mesh.bounds
        # Define the action space
        self.action_space_low = np.array([self.mesh_bbox[0][0] - sensor_range,
                                     self.mesh_bbox[0][1] - sensor_range,
                                     self.mesh_bbox[0][2] - sensor_range,
                                     -np.pi, -np.pi, -np.pi])
        self.action_space_high = np.array([self.mesh_bbox[1][0] + sensor_range,
                                      self.mesh_bbox[1][1] + sensor_range,
                                      self.mesh_bbox[1][2] + sensor_range,
                                      np.pi, np.pi, np.pi])

        self.action_space = spaces.Box(low=self.action_space_low, high=self.action_space_high)

        # find the number of mesh elements from the loaded mesh
        self.num_mesh_elements = len(self.mesh.faces)
        self.observation_space = spaces.Box(low=np.array([0]*self.num_mesh_elements), high=np.array([1]*self.num_mesh_elements), dtype=np.int32)
        self.observation_history = np.array([])

        # Extract vertices from the mesh
        self.vertices = self.mesh.vertices

        # Convert vertices to a list of tuples for easier handling
        self.mesh_elements = [(x, y, z) for x, y, z in self.vertices]

    def reset(self, seed=None, options=None):
        # Randomize the initial agent pose
        super().reset(seed=seed)
        self.agent_pose = self.action_space.sample()
        self.observation_history = np.array([])

    def step(self, action):

        self.agent_pose = action
        
        coverage_mask = self._compute_coverage_mask(self.agent_pose, self.sensor_range)
        self.observation_space = coverage_mask

        # Observation history is a 2D array with each row representing a single observation
        # Each observation is a 1D array with 0s and 1s representing the visibility of each mesh element
        self.observation_history = np.vstack((self.observation_history, self.observation_space)) if self.observation_history.size else np.reshape(np.append(self.observation_history,self.observation_space), (1, -1))

        # Compute reward based on the entire observation space
        reward = self._compute_reward(self.observation_space)

        done = self._is_episode_done()
        
        # Step returns observation of state, reward, done, and info in a tuple
        print(f"Reward: {reward}")
        return self.observation_space, reward, done, {}

    def _compute_coverage_mask(self, sensor_pose, sensor_info):
        face_centroids = np.zeros((self.num_mesh_elements, 3))
        for i, face in enumerate(self.mesh.faces):
            face_centroids[i] = np.mean(self.mesh.vertices[face], axis=0)

        # Calculate the distance between the sensor and the centroid of each face
        distances = np.linalg.norm(face_centroids - np.array(sensor_pose[:3]), axis=1)

        # Identify faces within the sensor range
        visible_faces = [face for face, distance in zip(self.mesh.faces, distances) if distance <= self.sensor_range]

        # Convert the indices of visible faces to a binary array (coverage mask)
        coverage_mask = np.zeros((self.num_mesh_elements))
        for face in visible_faces:
            # Find the index of the face in the mesh self.mesh.faces list
            face_index = np.where(self.mesh.faces == face)[0][0]
            coverage_mask[face_index] = 1

        return coverage_mask
    
    def _compute_reward(self, observation_space):
        # Extract the latest observation
        latest_observation = self.observation_history[-1, :]

        # Extract all elements that were visible in the past observations
        past_visible_elements = np.sum(self.observation_history[:-1, :], axis=0)

        # Calculate the set difference (elements visible in the current but not in the past)
        set_difference = np.sum(np.logical_and(latest_observation, np.logical_not(past_visible_elements)))

        # Reward based on the set difference
        reward = set_difference

        return reward
    
    def _is_episode_done(self):
        # Customize this method based on your termination condition
        # For example, you can check if the coverage goal is achieved or if the maximum number of steps is reached
        if self._check_termination():
            return True
        else:
            return False

    
    def _check_termination(self):
        # Calculate the total number of mesh elements
        total_mesh_elements = self.observation_history.shape[1]

        # Calculate the number of unique elements discovered across all observations
        unique_elements = np.unique(self.observation_history, axis=0)
        unique_elements_count = unique_elements.shape[0]

        # Check if more than 50% of the mesh elements have been discovered
        termination_condition = (unique_elements_count / total_mesh_elements) > 0.5

        return termination_condition
    
    def create_camera_mesh(self):
        # Define camera pyramid vertices and faces
        camera_vertices = np.array([[0, 0, 0], [-0.1, -0.1, 0.1], [0.1, -0.1, 0.1], [0.1, 0.1, 0.1], [-0.1, 0.1, 0.1]])
        camera_faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [1, 2, 3],[2, 3, 4]])

        # Create a trimesh object for the camera
        camera_mesh = trimesh.Trimesh(vertices=camera_vertices, faces=camera_faces)

        return camera_mesh

    def position_and_orient_camera(self):
        # Extract pose components
        camera_mesh = self.create_camera_mesh()
        x, y, z, alpha, beta, gamma = self.agent_pose

        # Define transformation matrix for translation and rotation
        transformation_matrix = trimesh.transformations.compose_matrix(
            translate=[x, y, z],
            angles=[alpha, beta, gamma],
            scale=[1, 1, 1]
        )

        # Apply the transformation to the camera mesh
        camera_mesh.apply_transform(transformation_matrix)

        return camera_mesh

    def render(self):
        # Find the sum of elements vertically and create a mask where the sum is more than 0. These are all elements visible
        latest_observation = np.sum(self.observation_history, axis=0)
        latest_observation = np.where(latest_observation > 0, 1, 0)
        print(f"How many faces covered: {np.sum(latest_observation)}")
        # print(f"No. of observations: {latest_observation.shape}")
        # print(latest_observation)

        test_mesh = self.mesh.copy()
        # print(f"No. of faces: {len(test_mesh.faces)}")
        for i, face in enumerate(test_mesh.faces):
            if latest_observation[i] == 1:
                test_mesh.visual.face_colors[face] = [255, 0, 0, 255]   # Red
            else:
                test_mesh.visual.face_colors[face] = [0, 255, 0, 255]   # Green

        # test_mesh.show()
        self.environment = test_mesh + self.position_and_orient_camera()

