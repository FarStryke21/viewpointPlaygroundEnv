# Write the setup.py file for the viewpointWorld-v0 environment.
# The setup.py file is used to install the viewpointWorld-v0 environment.

from setuptools import setup

setup(name='viewpointWorld-v0',
        version='0.0.1',
        install_requires=['gymnasium', 'numpy', 'trimesh', 'scipy', 'matplotlib']
    )

# Path: viewpoint_env/viewpointWorld-v0.py
# Write the viewpointWorld-v0 environment.
# The viewpointWorld-v0 environment is a 3D environment that is used to train a robot to explore a 3D environment.
# The robot is trained to explore the environment by maximizing the coverage of the environment.
# The environment is defined by a 3D mesh file.