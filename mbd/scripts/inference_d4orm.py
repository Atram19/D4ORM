import functools
import os
import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass
import tyro
from tqdm import tqdm

from mbd.envs import MultiCar2d
from mbd.utils import rollout_multi_us
from mbd.envs.multi_car import check_inter_robot_collisions
from mbd.planners.run_multicar import run_diffusion, Args

def run_diffusion_once(args: Args):
    """
    First phase of D4ORM: initial global reverse diffusion
    inspired by Algorithm 1 (Model-Based Diffusion) from the D4ORM paper.
    
    """

    rng = jax.random.PRNGKey(seed=args.seed)
    env = MultiCar2d(n=args.n_robots)

    Nx = env.observation_size
    Nu = env.action_size
    n = env.num_robots

    step_env_jit = jax.jit(env.step)
    reset_env_jit = jax.jit(env.reset)
    rollout_us = jax.jit(functools.partial(rollout_multi_us, step_env_jit))

    rng, rng_reset = jax.random.split(rng)
    state_init = reset_env_jit(rng_reset)

    # Diffusion noise schedule
    betas = jnp.linspace(args.beta0, args.betaT, 15)
    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)
    sigmas = jnp.sqrt(1 - alphas_bar)
   
    #  Start from zero control (like initial UN ~ N(0,I) in Algorithm 1)
    YN = jnp.zeros([args.Hsample, n, Nu])  

     # Single diffusion step
    @jax.jit
    def reverse_once(carry, _):
        i, rng, Ybar_i = carry
        Yi = Ybar_i * jnp.sqrt(alphas_bar[i])
       
        # Sample noisy controls
        rng, rng_eps = jax.random.split(rng)
        eps_u = jax.random.normal(rng_eps, (args.Nsample, args.Hsample, n, Nu))
        Y0s = eps_u * sigmas[i] + Ybar_i
        Y0s = jnp.clip(Y0s, -1.0, 1.0)
        
        rewss, _ = jax.vmap(rollout_us, in_axes=(None, 0))(state_init, Y0s)
        rews = rewss.mean(axis=(1, 2))
        
        rew_std = rews.std()
        rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std)
        logp0 = (rews - rews.mean()) / rew_std / args.temp_sample
      
        # Weighted average of samples  (Monte carlo estimate)
        weights = jax.nn.softmax(logp0)
        Ybar = jnp.einsum("s,shij->hij", weights, Y0s)
        # Reverse diffusion step
        score = 1 / (1.0 - alphas_bar[i]) * (-Yi + jnp.sqrt(alphas_bar[i]) * Ybar)
        Yim1 = 1 / jnp.sqrt(alphas[i]) * (Yi + (1.0 - alphas_bar[i]) * score)
        Ybar_im1 = Yim1 / jnp.sqrt(alphas_bar[i - 1])
        return (i - 1, rng, Ybar_im1), None

    def reverse(YN, rng):
        Yi = YN
        for i in reversed(range(1,15)):
            carry = (i, rng, Yi)
            (i, rng, Yi), _ = reverse_once(carry, None)
        return Yi  # U^(0)

    rng_exp, rng = jax.random.split(rng)
    U_0 = reverse(YN, rng_exp)

    # Rollout to get the trajectory associated at U_0
    xs = jnp.array([state_init.pipeline_state])
    state = state_init
    for t in range(U_0.shape[0]):
        state = step_env_jit(state, U_0[t])
        xs = jnp.concatenate([xs, state.pipeline_state[None]], axis=0)
    xs = jnp.transpose(xs, (1, 0, 2))  # (n, T+1, state_dim)
    
    return U_0, xs

def run_diffusion_local(args: Args, U_init: jnp.ndarray, trajectory_buffer: list, sample_buffer: list):
    """
    Second phase of D4ORM: local iterative reverse diffusion optimization.
    Based on Algorithm 2 (Iterative Denoising) from the D4ORM paper.
    
    Returns:
        U_opt: optimized control sequence
        trajectory_buffer: list of trajectories at each local diffusion step
    """
    rng = jax.random.PRNGKey(seed=args.seed + 123)  # different seed for local phase

    env = MultiCar2d(n=args.n_robots)
    step_env_jit = jax.jit(env.step)
    reset_env_jit = jax.jit(env.reset)
    rollout_us = jax.jit(functools.partial(rollout_multi_us, step_env_jit))

    H = args.Hsample
    Nu = env.action_size
    n = env.num_robots

    U = U_init.copy()

    # Buffers to save samples and trajectories
    sample_buffer = []
    trajectory_buffer = []

    # Local diffusion parameters
    L = 10  # window length
    K = 5   # number of local iterations

    betas_local = jnp.linspace(0.01, 0.2, 10)
    alphas_local = 1.0 - betas_local
    alphas_bar_local = jnp.cumprod(alphas_local)
    sigmas_local = jnp.sqrt(1 - alphas_bar_local)
    sigma_local = sigmas_local[-1]

    for k in range(K):
        for t_start in range(0, H - L + 1, L // 2):
            t_end = t_start + L
            U_window = U[t_start:t_end]

            rng, rng_step = jax.random.split(rng)

            def reverse_once_local(U_w, rng_w):
                eps_u = jax.random.normal(rng_w, (args.Nsample, L, n, Nu))
                Y0s = eps_u * sigma_local + U_w
                Y0s = jnp.clip(Y0s, -1.0, 1.0)

                U_fulls = jnp.repeat(U[None, ...], args.Nsample, axis=0)
                U_fulls = U_fulls.at[:, t_start:t_end, :, :].set(Y0s)

                state_init_local = reset_env_jit(rng_w)
                rewss, qss = jax.vmap(rollout_us, in_axes=(None, 0))(state_init_local, U_fulls)
                rews = rewss.mean(axis=(1, 2))

                logp0 = (rews - rews.mean()) / (rews.std() + 1e-6) / args.temp_sample
                weights = jax.nn.softmax(logp0)
                U_opt = jnp.einsum("s,slij->lij", weights, Y0s)

                return U_opt, qss

            U_opt_local, qss = reverse_once_local(U_window, rng_step)
            U = U.at[t_start:t_end].set(U_opt_local)

            # save the states obtained from the sampled actions
            sample_buffer.append((t_start, t_end, qss[:, t_start:t_end, :, :]))

            
            state = reset_env_jit(jax.random.PRNGKey(args.seed + 1000 + k))
            xs = jnp.array([state.pipeline_state])
            for t in range(U.shape[0]):
                state = step_env_jit(state, U[t])
                xs = jnp.concatenate([xs, state.pipeline_state[None]], axis=0)
            xs = jnp.transpose(xs, (1, 0, 2))  # shape: (n_robots, T, state_dim)
           
            # save the best trajectory locally optimized
            trajectory_buffer.append(xs)

    return U, trajectory_buffer, sample_buffer

def main():
    args = tyro.cli(Args)

    print("STEP 1: Initial Reverse Diffusion")
    U_init,traj_0 = run_diffusion_once(args)

    trajectory_buffer=[traj_0]
    sample_buffer= [None]
    print("STEP 2: Iterative Local Optimization with Visualization")
    U_opt, trajectory_buffer, sample_buffer = run_diffusion_local(args, U_init,trajectory_buffer,sample_buffer)

    if not args.not_render:
        path = "results/multicar_iterative"
        os.makedirs(path, exist_ok=True)

        fig, ax = plt.subplots(figsize=(5, 5))
        cmap = plt.get_cmap('tab20', args.n_robots)

        def init():
            ax.clear()
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_title("Local Diffusion Evolution")
            ax.set_aspect('equal')
            ax.grid(True)
            return []

        def update(frame_idx):
            ax.clear()
            # draw the sampled trajectories
            if sample_buffer[frame_idx] is not None:
                t_start, t_end, samples = sample_buffer[frame_idx]
                for i in range(args.n_robots):
                    color = cmap(i)
                    for k in range(min(80, samples.shape[0])):
                        traj = samples[k, :, i, :]
                        ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.1)
            # draw the optimized trajectory
            xs = trajectory_buffer[frame_idx]
            for i in range(args.n_robots):
                traj = xs[i]
                color = cmap(i)
                ax.plot(traj[:, 0], traj[:, 1], '-', color=color)
                ax.plot(traj[0, 0], traj[0, 1], 's', color=color, markersize=4)
                ax.plot(traj[-1, 0], traj[-1, 1], '*', color=color, markersize=7)

            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            title = "Initial Trajectory" if frame_idx == 0 else f"Diffusion Step {frame_idx}"
            ax.set_title(title)
            ax.set_aspect('equal')
            ax.grid(True)
            return []


        ani = animation.FuncAnimation(fig, update, frames=len(trajectory_buffer), init_func=init, blit=False)
        ani.save(os.path.join(path, "local_diffusion_video.mp4"), fps=10, dpi=150)
        print("Saved: local_diffusion_video.mp4")


if __name__ == "__main__":
    main()