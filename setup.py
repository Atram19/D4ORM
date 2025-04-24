from setuptools import setup, find_packages

setup(
    name='mbd',
    author="Chaoyi Pan",
    author_email="chaoyip@andrew.cmu.edu",
    packages=find_packages(include=["mbd"]),
    version='0.0.1',
    install_requires=[
        'brax',
        'gym', 
        'pandas', 
        'seaborn', 
        'matplotlib', 
        'imageio',
        'control', 
        'tqdm', 
        'tyro', 
        'meshcat', 
        'sympy', 
        'gymnax',
        'jax', 
        'distrax', 
        'gputil', 
        'jaxopt'
    ]
)