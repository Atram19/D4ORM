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

@dataclass
class Args:
    seed: int = 0
    n_robots: int = 4  
    Nsample: int = 2048     # numero di campioni
    Hsample: int = 100      # orizzonte
    Ndiffuse: int = 100     # numero di passi di diffusione
    temp_sample: float = 0.1 # temperatura
    beta0: float = 1e-4 
    betaT: float = 1e-2
    not_render: bool = False


def run_diffusion(args: Args, U_init=None, return_delta=False):
    rng = jax.random.PRNGKey(seed=args.seed)

    env = MultiCar2d(n=args.n_robots)

    Nx = env.observation_size
    Nu = env.action_size
    n = env.num_robots

    # per velocizzare 
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

    print(f"init sigma = {sigmas[-1]:.2e}")
    YN = jnp.zeros([args.Hsample, n, Nu]) if U_init is None else U_init

    @jax.jit
    def reverse_once(carry, unused):
        i, rng, Ybar_i = carry
        Yi = Ybar_i * jnp.sqrt(alphas_bar[i])

        rng, rng_eps = jax.random.split(rng)
        eps_u = jax.random.normal(rng_eps, (args.Nsample, args.Hsample, n, Nu))
        Y0s = eps_u * sigmas[i] + Ybar_i  
        Y0s = jnp.clip(Y0s, -1.0, 1.0)

        print("Y0s shape:", Y0s.shape)  

        # rollout
        rewss, qs = jax.vmap(rollout_us, in_axes=(None, 0))(state_init, Y0s)
        rews = rewss.mean(axis=(1, 2))


        # Normalize rewards → logp0
        rew_std = rews.std()
        rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std)
        rew_mean = rews.mean()
        logp0 = (rews - rew_mean) / rew_std / args.temp_sample

        weights = jax.nn.softmax(logp0)  # (Nsample,)
     
        Ybar = jnp.einsum("s,shij->hij", weights, Y0s)       # monte carlo (H, n, Nu)
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

    rng_exp, rng = jax.random.split(rng)
    Yi = reverse(YN, rng_exp)

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
        
        print("Controllo collisioni durante il rollout:")
        for t in range(xs.shape[1]):
            if check_inter_robot_collisions(xs[:, t, :], env.Ra):
                print(f" Collisione rilevata al timestep {t}")

        # print("Initial pipeline_state shape:", state_init.pipeline_state.shape)
        # print("Final xs shape (to render):", xs.shape)
        # if hasattr(env, "xg"):
        #     print("Goals shape:", env.xg.shape)
        # else:
        #     print("env has no attribute 'xg'")

        env.render(ax, xs, goals=env.xg)
        plt.savefig(f"{path}/rollout.png")
        # Matplotlib expects numpy arrays
        xs_np = jnp.array(xs)  # shape (n, T, 3)
        n, T, _ = xs_np.shape

        fig, ax = plt.subplots(figsize=(5, 5))
        lines = [ax.plot([], [], 'o', label=f"Robot {i}")[0] for i in range(n)]
        goals = env.xg if hasattr(env, "xg") else None

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_title("Robot rollout")
        ax.legend()

        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def update(frame):
            for i, line in enumerate(lines):
                x, y = xs_np[i, frame, 0], xs_np[i, frame, 1]
                line.set_data([x], [y])  
            return lines

        ani = animation.FuncAnimation(
            fig, update, frames=T, init_func=init, blit=True, interval=100
        )

        video_path = os.path.join(path, "rollout.mp4")
        ani.save(video_path, fps=10, dpi=150)
        print("Video salvato in:", video_path)


    # if return_delta and U_init is not None:
    #     return Yi[-1] - U_init
    # else:
    #     return Yi[-1]
    # Calcolo del vero reward medio finale
        # Dopo reverse diffusion
        
    delta_U = Yi[-1] - U_init if U_init is not None else Yi[-1]

    # Calcola reward vero della traiettoria ottimizzata
    U_opt = Yi[-1] if U_init is None else U_init + delta_U
    rewss_final, _ = jax.vmap(rollout_us, in_axes=(None, 0))(state_init, U_opt[None, ...])
    rew_per_robot = rewss_final[0].mean(axis=0)  # shape: (n,)

    return delta_U, rew_per_robot



if __name__ == "__main__":
    # rew_final = run_diffusion(args=tyro.cli(Args))
    # print("final reward =", jnp.mean(rew_final))

    final_trajectory, final_reward = run_diffusion(args=tyro.cli(Args))
    print(f"final reward per robot:",final_reward)