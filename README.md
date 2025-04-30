# D4ORM - Diffusion-based optimization for multi-robot trajectory planning

## Introduction

This repository builds upon the [Model-Based Diffusion for Trajectory Optimization (MBD)](https://arxiv.org/pdf/2407.01573) framework, extending it to support **multi-robot** trajectory optimization through a new algorithm called **D4ORM**(https://arxiv.org/pdf/2503.12204)

The original MBD approach was focused on optimizing trajectories for single agents in simulated environments. In D4ORM, we adapt the diffusion model framework to **coordinated multi-robot systems**.

---

## New Files

The following new files have been added to implement D4ORM:

- `multi_car.py` 
➔ Defines a **multi-robot 2D environment** where multiple robots move and avoid collisions.

- `run_multicar.py` 
➔ Runs the MBD **diffusion process** over multi-robot trajectories

- `iterative.py` 
➔ Implements the **D4ORM algorithm**

## Installation

Install the required packages by running:

```bash
git clone https://github.com/Atram19/D4ORM.git

pip install -e .

Make sure you have JAX and Optax installed to run the code.

If you encounter an error like:

jax._src.xla_bridge:909: An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.

you can fix it by installing JAX with CUDA support:

pip install "jax[cuda12]" optax


Usage

To run MBD optimization for multi-robot trajectories:

cd multi-robot-model-based-diffusion
python3 -m mbd.planners.run_multicar

To run the iterative D4ORM algorithm:

cd multi-robot-model-based-diffusion
python3 -m mbd.planners.iterative
