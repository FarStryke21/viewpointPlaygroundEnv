from setuptools import setup, find_packages

setup(
    nname='viewpoint-world',  # Use a valid package name
    version='0.0.1',
    packages=find_packages(),  # Automatically find packages in the directory
    install_requires=[
        'gymnasium',
        'numpy',
        'open3d==0.18.0',
        'scipy',
        'matplotlib'
    ],
    python_requires='>=3.6',
    description='A viewpoint world environment',  # Add a brief description
    author='Aman Chulawala',  # Add your name
    author_email='aman.chulawala@gmail.com',  # Add your email
)
