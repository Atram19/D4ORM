import functools
import os
import jax
from jax import numpy as jnp
from dataclasses import dataclass
import tyro
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.animation as animation

import mbd
from mbd.envs import MultiCar2d
from mbd.planners.run_multicar import Args

def run_diffusion(args: Args):
    rng = jax.random.PRNGKey(seed=args.seed)
    env = MultiCar2d(n=args.n_robots)

    Nx = env.observation_size
    Nu = env.action_size
    n = env.num_robots

    step_env_jit = jax.jit(env.step)
    reset_env_jit = jax.jit(env.reset)
    rollout_us = jax.jit(functools.partial(mbd.utils.rollout_multi_us, step_env_jit))

    rng, rng_reset = jax.random.split(rng)
    state_init = reset_env_jit(rng_reset)

    # Diffusion schedule
    betas = jnp.linspace(args.beta0, args.betaT, args.Ndiffuse)
    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)
    sigmas = jnp.sqrt(1 - alphas_bar)
    Sigmas_cond = (
        (1 - alphas) * (1 - jnp.sqrt(jnp.roll(alphas_bar, 1))) / (1 - alphas_bar)
    )
    sigmas_cond = jnp.sqrt(Sigmas_cond)
    sigmas_cond = sigmas_cond.at[0].set(0.0)


    YN = jnp.zeros([args.Hsample, n, Nu])
    # 
    trajectory_buffer = []
    sample_buffer = []

    @jax.jit
    def reverse_once(carry, unused):
        i, rng, Ybar_i = carry
        Yi = Ybar_i * jnp.sqrt(alphas_bar[i])

        # Sample noisy controls
        rng, rng_eps = jax.random.split(rng)
        eps_u = jax.random.normal(rng_eps, (args.Nsample, args.Hsample, n, Nu))
        Y0s = eps_u * sigmas[i] + Ybar_i
        Y0s = jnp.clip(Y0s, -1.0, 1.0)

        # Rollout with sampled controls
        rewss, qss = jax.vmap(rollout_us, in_axes=(None, 0))(state_init, Y0s)
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
        # we also return all samples of the states
        return (i - 1, rng, Ybar_im1), ( rews.mean(), qss)

    def reverse(YN, rng):
        Yi = YN
        Ybars = []
        with tqdm(range(args.Ndiffuse - 1, 0, -1), desc="Diffusing") as pbar:
            for i in pbar:
                carry_once = (i, rng, Yi)
                (i, rng, Yi), ( rew, qss) = reverse_once(carry_once, None)
                Ybars.append(Yi)
                # after each reverse step the evolution of the best trajectory is simulated
                state = state_init
                xs = jnp.array([state.pipeline_state])
                for t in range(Yi.shape[0]):
                    state = step_env_jit(state, Yi[t])
                    xs = jnp.concatenate([xs, state.pipeline_state[None]], axis=0)
                xs = jnp.transpose(xs, (1, 0, 2))
                
                trajectory_buffer.append(xs) # Save the best trajectory at each diffusion step

                sample_buffer.append(qss[:, :, :, :]) # Save the simulated noisy samples for visualization

                pbar.set_postfix({"rew": f"{rew:.2e}"})
        return jnp.array(Ybars)

    rng_exp, rng = jax.random.split(rng)
    Yi = reverse(YN, rng_exp)

    if not args.not_render:
        path = "results/multicar"
        os.makedirs(path, exist_ok=True)

        fig, ax = plt.subplots(figsize=(5, 5))
        cmap = plt.get_cmap('tab20', n)

        def init():
            ax.clear()
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_title("Diffusion Evolution")
            ax.set_aspect('equal')
            ax.grid(True)
            return []

        def update(frame_idx):
            ax.clear()
            # Draw the noisy samples
            samples = sample_buffer[frame_idx]  # (Nsample, H+1, n, 3)
            for i in range(n):
                color = cmap(i)
                for k in range(min(50, samples.shape[0])):  # max 50 samples
                    traj = samples[k, :, i, :]
                    ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.08) # alpha for transparency

            # Draw the best trajectory
            xs = trajectory_buffer[frame_idx]
            for i in range(n):
                traj = xs[i]
                color = cmap(i)
                ax.plot(traj[:, 0], traj[:, 1], '-', color=color)
                ax.plot(traj[0, 0], traj[0, 1], 's', color=color, markersize=4)
                ax.plot(traj[-1, 0], traj[-1, 1], '*', color=color, markersize=7)
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_title(f"Diffusion Step {frame_idx}")
            ax.set_aspect('equal')
            ax.grid(True)
            return []

        ani = animation.FuncAnimation(fig, update, frames=len(trajectory_buffer), init_func=init, blit=False)
        ani.save(os.path.join(path, "diffusion_video.mp4"), fps=10, dpi=150)
        print("Saved: diffusion_video.mp4")

    final = Yi[-1]
    rewss_final, _ = jax.vmap(rollout_us, in_axes=(None, 0))(state_init, final[None, ...])
    rew_per_robot = rewss_final[0].mean(axis=0)
    return final, rew_per_robot

if __name__ == "__main__":
    final_trajectory, final_reward = run_diffusion(args=tyro.cli(Args))
    print(f"Final reward for each robot: ", final_reward)
