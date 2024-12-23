import gymnasium as gym
import numpy as np
import torch
import open3d as o3d
from gymnasium import spaces
import copy
import matplotlib.pyplot as plt
import os

class CoverageEnv(gym.Env):
    def __init__(self, radius=20.0, coverage_threshold=0.98):
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
        self.state_visitation_counts = np.zeros(self.num_faces)

        # self.vis = o3d.visualization.Visualizer()
        # self.vis.create_window(visible=False)
        # self.vis.add_geometry(self.mesh)
        # self.ctr = self.vis.get_view_control()
        # self.camera = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
        self.actions = []
        self.reward_weights = np.array([100,     # w_newly_covered, 
                                       10.0,    # w_threshold_bonus, 
                                       -1.0,    # w_step_penalty, 
                                       10        # intrinsic reward
                                    ])

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.covered_faces.fill_(False)
        self.total_covered = 0
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        self.last_action = action  # Store the last action for rendering
        self.actions.append(action)
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
        reward = self._calculate_reward(newly_covered, coverage)
        done = coverage >= self.coverage_threshold # or self.steps >= 1000

        # print(f"Reward: {reward} | Action: {action} | Coverage: {self.total_covered / self.num_faces:.2%}")
        return self._get_obs(), reward, done, False, {}

    def _calculate_reward(self, newly_covered, coverage):
        # Calculate individual reward components
        r = np.array([
            torch.sum(newly_covered).item(),
            1 if coverage >= self.coverage_threshold else 0,
            len(self.actions),
            self._calculate_intrinsic_reward()
            ])
        # Combine reward components
        return np.dot(self.reward_weights, r)
    
    # TODO: explore intrinsic reward functions
    def _calculate_intrinsic_reward(self):
        current_state = self._get_obs()
        intrinsic_reward = 1 / (np.sqrt(self.state_visitation_counts[current_state]) + 1)
        self.state_visitation_counts[current_state] += 1
        return intrinsic_reward.mean()
    
    def _get_obs(self):
        return self.covered_faces.cpu().numpy()

    def get_mesh_file(self, folder):
        # Get the file path of the mesh file
        files = os.listdir(folder)
        # randomly select a file from the folder
        return np.random.choice(files)
        
    # Function when on the PC --------------
    def render(self):
        # Create a new visualizer for each render call
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Create a copy of the mesh for visualization
        vis_mesh = o3d.geometry.TriangleMesh()
        vis_mesh.vertices = o3d.utility.Vector3dVector(self.vertices.cpu().numpy())
        vis_mesh.triangles = o3d.utility.Vector3iVector(self.faces.cpu().numpy())

        # Color the mesh based on coverage
        colors = np.array([[0, 1, 0] if covered else [1, 0, 0] for covered in self.covered_faces.cpu().numpy()])
        vis_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        # Add the mesh to the visualizer
        vis.add_geometry(vis_mesh)

        # Create a sphere to represent the viewpoint
        viewpoint = o3d.geometry.TriangleMesh.create_sphere(radius=1)
        viewpoint.compute_vertex_normals()
        viewpoint.paint_uniform_color([1, 0, 0])  # Blue color for viewpoint

        # Set the position of the viewpoint
        # Move last_action to CPU before using with numpy
        last_action_cpu = self.last_action.cpu().numpy()
        theta = (last_action_cpu[0] + 1) * np.pi
        phi = last_action_cpu[1] * np.pi / 2
        x = self.radius * np.sin(theta) * np.cos(phi)
        y = self.radius * np.sin(theta) * np.sin(phi)
        z = self.radius * np.cos(theta)
        viewpoint.translate([x, y, z])

        # # Add the viewpoint to the visualizer
        vis.add_geometry(viewpoint)
        # o3d.visualization.draw_geometries([viewpoint, vis_mesh])

        # # Set up the camera view
        ctr = vis.get_view_control()
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(0.8)

        # # Update the geometry and render
        vis.update_geometry(vis_mesh)
        vis.update_geometry(viewpoint)
        vis.poll_events()
        vis.update_renderer()

        # Capture and display the image
        img = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()

        # Convert the image to numpy array and display using matplotlib
        plt.imshow(np.asarray(img))
        plt.title(f"Coverage: {self.total_covered / self.num_faces:.2%}")
        plt.axis('off')
        plt.show()

    # def render(self):
    #     try:
    #         # Create an OffscreenRenderer
    #         renderer = o3d.visualization.rendering.OffscreenRenderer(640, 480)
    #         print("1")
    #         # Create a copy of the mesh for visualization
    #         vis_mesh = o3d.geometry.TriangleMesh()
    #         vis_mesh.vertices = o3d.utility.Vector3dVector(self.vertices.cpu().numpy())
    #         vis_mesh.triangles = o3d.utility.Vector3iVector(self.faces.cpu().numpy())

    #         # Color the mesh based on coverage
    #         colors = np.array([[0, 1, 0] if covered else [1, 0, 0] for covered in self.covered_faces.cpu().numpy()])
    #         vis_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    #         print("2")
    #         # Add the mesh to the renderer
    #         renderer.scene.add_geometry("mesh", vis_mesh)
    #         print("3")
    #         # Create a sphere to represent the viewpoint
    #         viewpoint = o3d.geometry.TriangleMesh.create_sphere(radius=1)
    #         viewpoint.compute_vertex_normals()
    #         viewpoint.paint_uniform_color([0, 0, 1])  # Blue color for viewpoint
    #         print("4")
    #         # Set the position of the viewpoint
    #         last_action_cpu = self.last_action.cpu().numpy()
    #         theta = (last_action_cpu[0] + 1) * np.pi
    #         phi = last_action_cpu[1] * np.pi / 2
    #         x = self.radius * np.sin(theta) * np.cos(phi)
    #         y = self.radius * np.sin(theta) * np.sin(phi)
    #         z = self.radius * np.cos(theta)
    #         viewpoint.translate([x, y, z])
    #         print("5")
    #         # Add the viewpoint to the renderer
    #         renderer.scene.add_geometry("viewpoint", viewpoint)
    #         print("6")
    #         # Set up the camera view
    #         camera = o3d.camera.PinholeCameraParameters()
    #         camera.extrinsic = np.array([
    #             [1, 0, 0, 0],
    #             [0, 1, 0, 0],
    #             [0, 0, 1, -50],  # Move the camera back by 50 units
    #             [0, 0, 0, 1]
    #         ])
    #         camera.intrinsic.set_intrinsics(640, 480, 500, 500, 320, 240)
    #         renderer.setup_camera(camera.intrinsic, camera.extrinsic)
    #         print("7")
    #         # Render the scene
    #         img = renderer.render_to_image()
    #         print("8")
    #         # Convert the image to numpy array and display using matplotlib
    #         plt.imshow(np.asarray(img))
    #         plt.title(f"Coverage: {self.total_covered / self.num_faces:.2%}")
    #         plt.axis('off')
    #         plt.show()
    #     except Exception as e:
    #         print(e)

    def close(self):
        print(f"Coverage: {self.total_covered / self.num_faces:.2%}")
        if hasattr(self, 'vis'):
            self.vis.destroy_window()
