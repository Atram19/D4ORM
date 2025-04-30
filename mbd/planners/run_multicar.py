import functools
import os
import jax
from jax import numpy as jnp
from jax import config
from dataclasses import dataclass
import tyro
from tqdm import tqdm
from matplotlib import pyplot as plt

import mbd
from mbd.envs import MultiCar2d
from mbd.envs.multi_car import check_inter_robot_collisions
import matplotlib.animation as animation


# Define command-line arguments
@dataclass
class Args:
    seed: int = 0
    n_robots: int = 4
    Nsample: int = 2048         # number of samples
    Hsample: int = 100          # horizon
    Ndiffuse: int = 100         # number of diffusion steps
    temp_sample: float = 0.1    # temperature for sampling
    beta0: float = 1e-4         # initial noise
    betaT: float = 1e-2         # final noise
    not_render: bool = False
    high_resolution: bool = False

# Diffusion-based optimization process for multi-robot trajectory planning
def run_diffusion(args: Args):

    rng = jax.random.PRNGKey(seed=args.seed)
    env = MultiCar2d(n=args.n_robots)

    Nx = env.observation_size
    Nu = env.action_size
    n = env.num_robots

    # JIT-compile step, reset, and rollout functions to speed up execution
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

    # initialize diffusion process
    YN = jnp.zeros([args.Hsample, n, Nu])

    # Single diffusion step
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
        rewss, qs = jax.vmap(rollout_us, in_axes=(None, 0))(state_init, Y0s)
        rews = rewss.mean(axis=(1, 2))


        # Normalize rewards → logp0
        rew_std = rews.std()
        rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std)
        rew_mean = rews.mean()
        logp0 = (rews - rew_mean) / rew_std / args.temp_sample

        # Weighted average of samples  (Monte carlo estimate)
        weights = jax.nn.softmax(logp0)
        Ybar = jnp.einsum("s,shij->hij", weights, Y0s)

        # Reverse diffusion step
        score = 1 / (1.0 - alphas_bar[i]) * (-Yi + jnp.sqrt(alphas_bar[i]) * Ybar)
        Yim1 = 1 / jnp.sqrt(alphas[i]) * (Yi + (1.0 - alphas_bar[i]) * score)
        Ybar_im1 = Yim1 / jnp.sqrt(alphas_bar[i - 1])

        return (i - 1, rng, Ybar_im1), rews.mean()


    def reverse(YN, rng):
        Yi = YN
        Ybars = []
        with tqdm(range(args.Ndiffuse - 1, 0, -1), desc="Diffusing") as pbar:
            for i in pbar:
                carry_once = (i, rng, Yi)
                (i, rng, Yi), rew = reverse_once(carry_once, None)
                Ybars.append(Yi)
                pbar.set_postfix({"rew": f"{rew:.2e}"})
        return jnp.array(Ybars)

    # Run reverse diffusion
    rng_exp, rng = jax.random.split(rng)
    Yi = reverse(YN, rng_exp)

    # linearly interpolate 
    def interpolate_trajectory_jax(xs: jnp.ndarray, dt_original: float = 0.1, dt_interp: float = 0.01):
        n, T, d = xs.shape
        t_max = (T - 1) * dt_original
        t_interp = jnp.arange(0.0, t_max + dt_interp, dt_interp)  # (T_interp,)

        t_original = jnp.arange(0.0, T * dt_original, dt_original)  

        def interp_single_robot(traj):  # traj: (T, d)
            def interpolate_one(ti):
                idx = jnp.floor(ti / dt_original).astype(int)   
                idx = jnp.clip(idx, 0, T - 2)
                t0 = t_original[idx]
                t1 = t_original[idx + 1]
                x0 = traj[idx]
                x1 = traj[idx + 1]
                alpha = (ti - t0) / (t1 - t0)
                return (1 - alpha) * x0 + alpha * x1

            return jax.vmap(interpolate_one)(t_interp)  # (T_interp, d)

        xs_interp = jax.vmap(interp_single_robot)(xs)  # (n, T_interp, d)
        return xs_interp, t_interp
   


    if not args.not_render:
        
        path = "results/multicar"
        os.makedirs(path, exist_ok=True)

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        xs = jnp.array([state_init.pipeline_state])
        state = state_init
        for t in range(Yi.shape[1]):
            state = step_env_jit(state, Yi[-1, t])
            xs = jnp.concatenate([xs, state.pipeline_state[None]], axis=0)
        xs = jnp.transpose(xs, (1, 0, 2))  # shape: (n, H+1, 3)

        print("Check collisions during rollout:")
        for t in range(xs.shape[1]):
            if check_inter_robot_collisions(xs[:, t, :], env.Ra):
                print(f"Collision detected at timestep {t}")
        
        # === Interpolation ===
        if args.high_resolution:
            print("Interpolating high resolution trajectory for smoother rendering...")
            xs_interp, t_interp = interpolate_trajectory_jax(xs, dt_original=0.1, dt_interp=0.01)
            xs_np = jnp.array(xs_interp)
        else:
            print("Using original resolution trajectory for rendering...")
            xs_np = jnp.array(xs)


        n, T, _ = xs_np.shape

        fig_static, ax_static = plt.subplots(figsize=(5, 5))
        env.render(ax_static, xs, goals=env.xg)
        plt.title("Optimized final trajectory")
        plt.tight_layout()
        plt.savefig(f"{path}/global_diffusion.png")
        print(f"Figure saved in {path}/global_diffusion.png")


        fig, ax = plt.subplots(figsize=(5, 5))
        lines = [ax.plot([], [], 'o', label=f"Robot {i}")[0] for i in range(n)]
        goals = env.xg if hasattr(env, "xg") else None
        


        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_title("Robot tracking")
        ax.legend()
       
        # Create line and point objects for each robot
        lines = []
        points = []
       
        cmap = plt.get_cmap('tab20', n)
        for i in range(n):
            color = cmap(i)
            line, = ax.plot([], [], lw=2,color = color)
            lines.append(line)


            point, = ax.plot([], [], 'o', markersize=6,color = color)
            points.append(point)

            if goals is not None:
                    gx, gy = goals[i, 0], goals[i, 1]
                    ax.plot(gx, gy, 's', color = color, markersize=6, markeredgewidth=2)
    
           
        def init():
            for line, point in zip(lines, points):
                 line.set_data([], [])
                 point.set_data([], [])
            return lines + points 
        
        def update(frame):
            for i in range(n):
                x_trail = xs_np[i, :frame + 1, 0]
                y_trail = xs_np[i, :frame + 1, 1]
                lines[i].set_data(x_trail, y_trail)

                x_curr = xs_np[i, frame, 0]
                y_curr = xs_np[i, frame, 1]
                points[i].set_data([x_curr], [y_curr])
            return lines + points

        ani = animation.FuncAnimation(
            fig, update, frames=T, init_func=init, blit=False, interval=100
        )
        labels = [f"Robot {i}" for i in range(n)]
        ax.legend(points, labels, loc='upper right')

        video_path = os.path.join(path, "global_diffusion.mp4")
        ani.save(video_path, fps=10, dpi=150)
        print("Video saved in:", video_path)

    final = Yi[-1]

    # Calculate reward of the optimized trajectory
    rewss_final, _ = jax.vmap(rollout_us, in_axes=(None, 0))(state_init, final[None, ...])
    rew_per_robot = rewss_final[0].mean(axis=0)  

    return final, rew_per_robot



if __name__ == "__main__":

    final_trajectory, final_reward = run_diffusion(args=tyro.cli(Args))
    print(f"Final reward for each robot: ",final_reward)