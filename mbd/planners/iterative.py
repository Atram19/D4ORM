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

# Single-pass reverse diffusion to initialize U
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
    return U_0

# Local iterative diffusion optimization
def run_diffusion_local(args: Args, U_init: jnp.ndarray):
    """
    Second phase of D4ORM: local iterative reverse diffusion optimization.
    Based on Algorithm 2 (Iterative Denoising) from the D4ORM paper.
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

    # Local diffusion parameters
    L = 10  # window length
    K = 5   # number of local iterations

    
    # Local diffusion schedule (more noisy)
    betas_local = jnp.linspace(0.01, 0.2, 10)
    alphas_local = 1.0 - betas_local
    alphas_bar_local = jnp.cumprod(alphas_local)
    sigmas_local = jnp.sqrt(1 - alphas_bar_local)

    sigma_local = sigmas_local[-1]  # # max local noise
    


    for k in range(K):
        for t_start in range(0, H - L + 1, L // 2):  # sliding overlapping windows
            t_end = t_start + L
            U_window = U[t_start:t_end]  # shape: (L, n, Nu)

            rng, rng_step = jax.random.split(rng)

            # Local reverse diffusion inside the window
            def reverse_once_local(U_w, rng_w):
                eps_u = jax.random.normal(rng_w, (args.Nsample, L, n, Nu))
               # Y0s = eps_u * sigmas[-1] + U_w
                Y0s = eps_u * sigma_local + U_w
                Y0s = jnp.clip(Y0s, -1.0, 1.0)

                # Insert modified window into full control sequences
                U_fulls = jnp.repeat(U[None, ...], args.Nsample, axis=0)  # (Nsample, H, n, Nu)
                U_fulls = U_fulls.at[:, t_start:t_end, :, :].set(Y0s)
                #  Evaluate new rollouts
                state_init = reset_env_jit(rng_step)
                rewss, _ = jax.vmap(rollout_us, in_axes=(None, 0))(state_init, U_fulls)
                rews = rewss.mean(axis=(1, 2))

                # Compute weighted average of samples
                logp0 = (rews - rews.mean()) / (rews.std() + 1e-6) / args.temp_sample
                weights = jax.nn.softmax(logp0)
                U_opt = jnp.einsum("s,slij->lij", weights, Y0s)
                 
                return U_opt

            U_opt_local = reverse_once_local(U_window, rng_step)
            U = U.at[t_start:t_end].set(U_opt_local)
            # Print mean reward per robot after window optimization
            state_init_eval = reset_env_jit(jax.random.PRNGKey(args.seed + 999 + k))
            rewss_eval, _ = rollout_us(state_init_eval, U)
            # reward_mean = rewss_eval.mean()
            # print(f"[Iterazione {k}] Reward medio attuale: {reward_mean:.4f}")
            reward_per_robot = rewss_eval.mean(axis=0)
            reward_array_str = "[" + ", ".join(f"{r:.4f}" for r in reward_per_robot) + "]"
            print(f"[Iteration {k}] robots average rewards: {reward_array_str}")

    return U

def main():
    args = tyro.cli(Args)

    print("STEP 1: Initial Reverse Diffusion")
    U_init = run_diffusion_once(args)

    print("STEP 2: Iterative Local Optimization")
    U_opt = run_diffusion_local(args, U_init)

    print("STEP 3: Rollout with optimized controls")
    env = MultiCar2d(n=args.n_robots)
    step_env_jit = jax.jit(env.step)
    reset_env_jit = jax.jit(env.reset)
    rollout_us = jax.jit(lambda state, us: rollout_multi_us(step_env_jit, state, us))

    state_init = reset_env_jit(jax.random.PRNGKey(args.seed + 999))
    _, traj = rollout_us(state_init, U_opt)
    traj = jnp.concatenate([state_init.pipeline_state[None], traj], axis=0) # (T,n,3)
    traj = jnp.transpose(traj, (1, 0, 2))  # (n, T, 3)
    

    print("Check collisions during rollout:")
    for t in range(traj.shape[1]):
        if check_inter_robot_collisions(traj[:, t, :], env.Ra):
            print(f"Collision detected at timestep {t}")
   
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
        path = "results/multicar_iterative"
        os.makedirs(path, exist_ok=True)

    
        state_init = reset_env_jit(jax.random.PRNGKey(args.seed + 999))
        xs = jnp.array([state_init.pipeline_state])
        state = state_init
        for t in range(U_opt.shape[0]):
            state = step_env_jit(state, U_opt[t])
            xs = jnp.concatenate([xs, state.pipeline_state[None]], axis=0)
        xs = jnp.transpose(xs, (1, 0, 2))  

        # Salva immagine finale
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        env.render(ax, xs, goals=env.xg)
        plt.title("Optimized final trajectory(D4ORM)")
        plt.tight_layout()
        plt.savefig(os.path.join(path, "local_diffusion.png"))
        print(f"Figura salvata in {path}/local_diffusion.png")

        # Create video
        # === Interpolation ===
        if args.high_resolution:
            print("Interpolating high resolution trajectory for smoother rendering...")
            xs_interp, t_interp = interpolate_trajectory_jax(xs, dt_original=0.1, dt_interp=0.01)
            xs_np = jnp.array(xs_interp)
        else:
            print("Using original resolution trajectory for rendering...")
            xs_np = jnp.array(xs)
        
        n, T, _ = xs_np.shape
        cmap = plt.get_cmap('tab20', n)

        fig, ax = plt.subplots(figsize=(5, 5))
        lines = [ax.plot([], [], 'o', label=f"Robot {i}")[0] for i in range(n)]
        goals = env.xg if hasattr(env, "xg") else None

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_title("Robot tracking")
        ax.legend()

        def init():
            for line in lines:
                line.set_data([], [])
            return lines
        
        # Create line and point objects for each robot
        lines = []     
        points = []    
        
        for i in range(n):  
            color = cmap(i)
            line, = ax.plot([], [], lw=2,color = color) 
            lines.append(line)

            
            point, = ax.plot([], [], 'o', markersize=6,color = color)  
            points.append(point)

            if goals is not None:
                    gx, gy = goals[i, 0], goals[i, 1]
                    ax.plot(gx, gy, 's', color=color, markersize=6, markeredgewidth=2)
       
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


        video_path = os.path.join(path, "local_diffusion.mp4")
        ani.save(video_path, fps=10, dpi=150)
        print("Video saved in:", video_path)



if __name__ == "__main__":
    main()
