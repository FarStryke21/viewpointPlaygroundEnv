import gymnasium as gym
import numpy as np
import torch
import open3d as o3d
from gymnasium import spaces
import copy
import matplotlib.pyplot as plt
import os

class CoverageEnv(gym.Env):
    def __init__(self,
                radius=1.0, coverage_threshold=0.95):
        super(CoverageEnv, self).__init__()

        mesh_folder='/home/dir/RL_CoveragePlanning/test_models/modified'
        self.mesh_file_name = self.get_mesh_file(mesh_folder)
        print(f"Using {self.mesh_file_name}")
        obj_file_path = os.path.join(mesh_folder, self.mesh_file_name)
        
        self.mesh = o3d.io.read_triangle_mesh(obj_file_path)
        self.mesh.compute_vertex_normals()
        
        self.vertices = torch.tensor(np.asarray(self.mesh.vertices), dtype=torch.float32).cuda()
        self.faces = torch.tensor(np.asarray(self.mesh.triangles), dtype=torch.int64).cuda()
        self.normals = torch.tensor(np.asarray(self.mesh.vertex_normals), dtype=torch.float32).cuda()
        
        self.num_faces = self.faces.shape[0]
        self.radius = radius
        self.coverage_threshold = coverage_threshold

        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_faces,), dtype=np.int8)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        self.covered_faces = torch.zeros(self.num_faces, dtype=torch.bool).cuda()
        self.total_covered = 0
        self.steps = 0

        # self.vis = o3d.visualization.Visualizer()
        # self.vis.create_window(visible=False)
        # self.vis.add_geometry(self.mesh)
        # self.ctr = self.vis.get_view_control()
        # self.camera = self.vis.get_view_control().convert_to_pinhole_camera_parameters()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.covered_faces.fill_(False)
        self.total_covered = 0
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        self.last_action = action  # Store the last action for rendering
        self.steps += 1
        # The agent currently is returning actions as a numpy, we convert them to GPU Tensors
        # TODO: Look into getting tensors directly from agent (entire process on GPU, not just the environment)
        action = torch.tensor(action, dtype=torch.float32).cuda()
        theta = (action[0] + 1) * np.pi
        phi = action[1] * np.pi / 2

        x = self.radius * torch.sin(theta) * torch.cos(phi)
        y = self.radius * torch.sin(theta) * torch.sin(phi)
        z = self.radius * torch.cos(theta)
        viewpoint = torch.tensor([x, y, z], dtype=torch.float32).cuda()

        face_centers = torch.mean(self.vertices[self.faces], dim=1)
        face_normals = torch.cross(self.vertices[self.faces[:, 1]] - self.vertices[self.faces[:, 0]],
                                   self.vertices[self.faces[:, 2]] - self.vertices[self.faces[:, 0]])
        face_normals = face_normals / torch.norm(face_normals, dim=1, keepdim=True)

        view_directions = face_centers - viewpoint
        view_directions = view_directions / torch.norm(view_directions, dim=1, keepdim=True)

        visible_faces = torch.sum(face_normals * view_directions, dim=1) < 0

        newly_covered = visible_faces & ~self.covered_faces
        self.covered_faces |= visible_faces
        self.total_covered = torch.sum(self.covered_faces).item()

        coverage = self.total_covered / self.num_faces
        reward = torch.sum(newly_covered).item() * 0.1
        if coverage >= self.coverage_threshold:
            reward += 10
        reward -= 0.01

        done = coverage >= self.coverage_threshold or self.steps >= 1000

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        return self.covered_faces.cpu().numpy()
    
    def get_mesh_file(self, folder):
        # Get the file path of the mesh file
        files = os.listdir(folder)
        # randomly select a file from the folder
        return np.random.choice(files)

    def render(self):
        # Update mesh colors based on coverage
        colors = np.array([[0, 1, 0] if covered else [1, 0, 0] for covered in self.covered_faces.cpu().numpy()])
        self.mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        self.vis.update_geometry(self.mesh)

        # Set camera position based on the last action
        theta = (self.last_action[0] + 1) * np.pi
        phi = self.last_action[1] * np.pi / 2
        x = self.radius * np.sin(theta) * np.cos(phi)
        y = self.radius * np.sin(theta) * np.sin(phi)
        z = self.radius * np.cos(theta)
        
        self.camera.extrinsic = np.array([
            [1, 0, 0, -x],
            [0, 1, 0, -y],
            [0, 0, 1, -z],
            [0, 0, 0, 1]
        ])
        self.ctr.convert_from_pinhole_camera_parameters(self.camera)

        # Render image
        self.vis.poll_events()
        self.vis.update_renderer()
        image = self.vis.capture_screen_float_buffer(do_render=True)

        # Display image using matplotlib
        plt.figure(figsize=(10, 10))
        plt.imshow(np.asarray(image))
        plt.title(f"Coverage: {self.total_covered / self.num_faces:.2%}")
        plt.axis('off')
        plt.show()

    def close(self):
        self.vis.destroy_window()