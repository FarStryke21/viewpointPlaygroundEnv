# Environemt for Coverage Viewpoint Generation for a given mesh

## Install instructions

Navigate into the root directory of the repository and perform a source install

```
pip install -e .
```

## Usage

```
from viewpoint_env.viewpointWorld import CoverageEnv

env = CoverageEnv(path_to_mesh_file)
env.reset()
```


## Description

This project is a custom reinforcement learning environment, named `CoverageEnv`, built using the OpenAI Gym interface. The environment represents a 3D space with a mesh surface, and an agent that can move around to observe the mesh. The goal of the agent is to cover as much of the mesh as possible within its sensor range. The project includes methods for computing the coverage mask, reward calculation, checking termination conditions, and rendering the environment for visualization.

**Observation Space:** The observation space is a binary vector with length equal to the number of mesh faces. Face elements visible in a state are marked 1 and the remaining are 0.

**Observation History:** An array containing past observations.

**Action Space:** The area where the camera can be allowed. At present it is defined as a bounded box around the mesh whose dimensions are at an offset of sensor range from the actual bounding box from the mesh. The action space at present allows for regions within the mesh as well. Needs to be redefined.
