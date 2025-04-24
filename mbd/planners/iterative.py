import functools
import jax
from jax import numpy as jnp
from mbd.envs import MultiCar2d
import os
from mbd.utils import rollout_multi_us  
from mbd.planners.run_multicar import run_diffusion, Args
from mbd.envs.multi_car import check_inter_robot_collisions
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mbd.envs import MultiCar2d
from mbd.utils import rollout_multi_us
import tyro

def run_diffusion_once(args: Args):

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

    # Diffusion schedule
    betas = jnp.linspace(args.beta0, args.betaT, 5)
    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)
    sigmas = jnp.sqrt(1 - alphas_bar)

    YN = jnp.zeros([args.Hsample, n, Nu])  # invece di campionare dalla gaussiana si parte da un controllo tutto nullo


    def reverse_once(carry, _):
        i, rng, Ybar_i = carry
        Yi = Ybar_i * jnp.sqrt(alphas_bar[i])
        # campionamento delle traiettorie
        rng, rng_eps = jax.random.split(rng)
        eps_u = jax.random.normal(rng_eps, (args.Nsample, args.Hsample, n, Nu))
        Y0s = eps_u * sigmas[i] + Ybar_i
        Y0s = jnp.clip(Y0s, -1.0, 1.0)
        # rollout
        rewss, _ = jax.vmap(rollout_us, in_axes=(None, 0))(state_init, Y0s)
        rews = rewss.mean(axis=(1, 2))
        # softmax pesi
        rew_std = rews.std()
        rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std)
        logp0 = (rews - rews.mean()) / rew_std / args.temp_sample
        weights = jax.nn.softmax(logp0)
    
        Ybar = jnp.einsum("s,shij->hij", weights, Y0s)
        score = 1 / (1.0 - alphas_bar[i]) * (-Yi + jnp.sqrt(alphas_bar[i]) * Ybar)
        Yim1 = 1 / jnp.sqrt(alphas[i]) * (Yi + (1.0 - alphas_bar[i]) * score)
        Ybar_im1 = Yim1 / jnp.sqrt(alphas_bar[i - 1])
        return (i - 1, rng, Ybar_im1), None

    def reverse(YN, rng):
        Yi = YN
        for i in reversed(range(1,5)):
            carry = (i, rng, Yi)
            (i, rng, Yi), _ = reverse_once(carry, None)
        return Yi  # U^(0)

    rng_exp, rng = jax.random.split(rng)
    U_0 = reverse(YN, rng_exp)
    return U_0

def run_diffusion_local(args: Args, U_init: jnp.ndarray):
   

    rng = jax.random.PRNGKey(seed=args.seed + 123)  # nuovo seme per local

    env = MultiCar2d(n=args.n_robots)
    step_env_jit = jax.jit(env.step)
    reset_env_jit = jax.jit(env.reset)
    rollout_us = jax.jit(functools.partial(rollout_multi_us, step_env_jit))

    H = args.Hsample
    Nu = env.action_size
    n = env.num_robots

    U = U_init.copy()

    L = 10  # lunghezza finestra
    K = 5   # numero iterazioni locali

    # # Schedulazione diffusione (la stessa di run_diffusion_once)
    # betas = jnp.linspace(args.beta0, args.betaT, args.Ndiffuse)
    # alphas = 1.0 - betas
    # alphas_bar = jnp.cumprod(alphas)
    # sigmas = jnp.sqrt(1 - alphas_bar)
    # Schedulazione locale: più rumorosa e corta
    betas_local = jnp.linspace(0.01, 0.2, 10)
    alphas_local = 1.0 - betas_local
    alphas_bar_local = jnp.cumprod(alphas_local)
    sigmas_local = jnp.sqrt(1 - alphas_bar_local)

    sigma_local = sigmas_local[-1]  # massimo rumore della schedulazione locale
    


    for k in range(K):
        for t_start in range(0, H - L + 1, L // 2):  # finestre sovrapposte
            t_end = t_start + L
            U_window = U[t_start:t_end]  # shape: (L, n, Nu)

            rng, rng_step = jax.random.split(rng)

            # Una reverse diffusion SOLO su questa finestra
            def reverse_once_local(U_w, rng_w):
                eps_u = jax.random.normal(rng_w, (args.Nsample, L, n, Nu))
               # Y0s = eps_u * sigmas[-1] + U_w
                Y0s = eps_u * sigma_local + U_w
                Y0s = jnp.clip(Y0s, -1.0, 1.0)

                # Ricostruisci controllo globale con finestra modificata
                U_fulls = jnp.repeat(U[None, ...], args.Nsample, axis=0)  # (Nsample, H, n, Nu)
                U_fulls = U_fulls.at[:, t_start:t_end, :, :].set(Y0s)

                state_init = reset_env_jit(rng_step)
                rewss, _ = jax.vmap(rollout_us, in_axes=(None, 0))(state_init, U_fulls)
                rews = rewss.mean(axis=(1, 2))

                # Softmax pesi
                logp0 = (rews - rews.mean()) / (rews.std() + 1e-6) / args.temp_sample
                weights = jax.nn.softmax(logp0)
                U_opt = jnp.einsum("s,slij->lij", weights, Y0s)
                 
                return U_opt

            U_opt_local = reverse_once_local(U_window, rng_step)
            U = U.at[t_start:t_end].set(U_opt_local)
            # Calcolo reward medio sull'intera traiettoria aggiornata
            state_init_eval = reset_env_jit(jax.random.PRNGKey(args.seed + 999 + k))
            rewss_eval, _ = rollout_us(state_init_eval, U)
            # reward_mean = rewss_eval.mean()
            # print(f"[Iterazione {k}] Reward medio attuale: {reward_mean:.4f}")
            reward_per_robot = rewss_eval.mean(axis=0)
            reward_array_str = "[" + ", ".join(f"{r:.4f}" for r in reward_per_robot) + "]"
            print(f"[Iterazione {k}] Reward per robot: {reward_array_str}")

    return U

def main():
    args = tyro.cli(Args)

    print("STEP 1: Reverse Diffusion iniziale")
    U_init = run_diffusion_once(args)

    print("STEP 2: Ottimizzazione Locale Iterativa")
    U_opt = run_diffusion_local(args, U_init)

    print("STEP 3: Rollout finale")
    env = MultiCar2d(n=args.n_robots)
    step_env_jit = jax.jit(env.step)
    reset_env_jit = jax.jit(env.reset)
    rollout_us = jax.jit(lambda state, us: rollout_multi_us(step_env_jit, state, us))

    state_init = reset_env_jit(jax.random.PRNGKey(args.seed + 999))
    _, traj = rollout_us(state_init, U_opt)
    traj = jnp.concatenate([state_init.pipeline_state[None], traj], axis=0) # (T,n,3)
    traj = jnp.transpose(traj, (1, 0, 2))  # (n, T, 3)

    # Check collisioni
    print("Controllo collisioni finali:")
    for t in range(traj.shape[1]):
        if check_inter_robot_collisions(traj[:, t, :], env.Ra):
            print(f" Collisione al timestep {t}")

    # # Rendering finale (solo se non disattivato da args)
    # if not args.not_render:
    #     path = "results/multicar_iterative"
    #     os.makedirs(path, exist_ok=True)

    #     fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    #     env.render(ax, traj, goals=env.xg)
    #     plt.title("Traiettoria finale ottimizzata (D4ORM)")
    #     plt.tight_layout()
    #     plt.savefig(f"{path}/rollout.png")
    #     print(f" Figura salvata in {path}/rollout.png")
    if not args.not_render:
        path = "results/multicar_iterative"
        os.makedirs(path, exist_ok=True)

        # Rollout finale
        state_init = reset_env_jit(jax.random.PRNGKey(args.seed + 999))
        xs = jnp.array([state_init.pipeline_state])
        state = state_init
        for t in range(U_opt.shape[0]):
            state = step_env_jit(state, U_opt[t])
            xs = jnp.concatenate([xs, state.pipeline_state[None]], axis=0)
        xs = jnp.transpose(xs, (1, 0, 2))  # shape: (n, H+1, 3)

        # Salva immagine finale
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        env.render(ax, xs, goals=env.xg)
        plt.title("Traiettoria finale ottimizzata (D4ORM)")
        plt.tight_layout()
        plt.savefig(os.path.join(path, "rollout.png"))
        print(f"Figura salvata in {path}/rollout.png")

        # Crea video animato
        xs_np = jnp.array(xs)
        n, T, _ = xs_np.shape

        fig, ax = plt.subplots(figsize=(5, 5))
        lines = [ax.plot([], [], 'o', label=f"Robot {i}")[0] for i in range(n)]
        goals = env.xg if hasattr(env, "xg") else None

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_title("Robot rollout (video)")
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



if __name__ == "__main__":
    main()
