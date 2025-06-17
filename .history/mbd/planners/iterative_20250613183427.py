import functools
import jax
from jax import numpy as jnp

import os
from mbd.utils import rollout_multi_us , make_lagrangian_fn, make_residual_fn
from mbd.envs.multi_car import check_inter_robot_collisions, Args
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mbd.envs import MultiCar2d

import tyro
import numpy as np
import time
# Single-pass reverse diffusion to initialize U
def run_diffusion_once(args: Args,env, rollout_us, reset_env_jit):
    """
    First phase of D4ORM: initial global reverse diffusion
    inspired by Algorithm 1 (Model-Based Diffusion) from the D4ORM paper.

    """
    rng = jax.random.PRNGKey(seed=args.seed)
 
    Nx = env.observation_size
    Nu = env.action_size
    n = env.num_robots

    def exponential_beta_schedule(n_diffusion_steps, beta_start=1e-4, beta_end=1.0):
        x = jnp.linspace(0, n_diffusion_steps - 1, n_diffusion_steps)
        log_ratio = jnp.log(beta_end / beta_start)
        return beta_start * jnp.exp(log_ratio * x / n_diffusion_steps)

    rng, rng_reset = jax.random.split(rng)
    state_init = reset_env_jit(rng_reset)
    if args.save_video: 
        Yi_list = []
        Y0s_list = []
        trajectories_denoised = []
        trajectories_samples = []

    # Diffusion noise schedule
    #betas = jnp.linspace(args.beta0, args.betaT, 100)
    betas = exponential_beta_schedule(n_diffusion_steps=100, beta_start=args.beta0, beta_end=args.betaT)

    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)
    sigmas = jnp.sqrt(1 - alphas_bar)* args.noise_scale

    #  Start from zero control
    #YN = jnp.zeros([args.Hsample, n, Nu])
    eps_init = jax.random.normal(rng, shape=(args.Hsample, n, Nu))
    YN = eps_init * 3


     # Single diffusion step
    #@jax.jit
    def reverse_once(carry):
        i, rng, Ybar_i = carry
        Yi = Ybar_i * jnp.sqrt(alphas_bar[i])

        # Sample noisy controls
        rng, rng_eps = jax.random.split(rng)
        eps_u = jax.random.normal(rng_eps, (args.Nsample, args.Hsample, n, Nu))
        Y0s = eps_u * sigmas[i] + Ybar_i
        #Y0s = jnp.clip(Y0s, -1.0, 1.0)

        rewss, _ = jax.vmap(rollout_us, in_axes=(None, 0))(state_init, Y0s)
        rews = rewss.mean(axis=(1, 2))

        rew_std = rews.std()
        rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std)
        logp0 = (rews - rews.mean()) / rew_std / args.temp_sample
        if i % 10 == 0:
            # Debug: range dei reward
            print("Reward min/max:", rews.min(), rews.max())
            # Debug: varianza dei sample (quanto i controlli sono diversi tra loro)
            print("Varianza Y0s:", Y0s.var())
            # Debug: media campioni (vedi se collassano a zero)
            print("Mean Y0s:", Y0s.mean())
        # Weighted average of samples  (Monte carlo estimate)
        weights = jax.nn.softmax(logp0)
        Ybar = jnp.einsum("s,shij->hij", weights, Y0s)
        # Reverse diffusion step
        score = 1 / (1.0 - alphas_bar[i]) * (-Yi + jnp.sqrt(alphas_bar[i]) * Ybar)
        Yim1 = 1 / jnp.sqrt(alphas[i]) * (Yi + (1.0 - alphas_bar[i]) * score)
        Ybar_im1 = Yim1 / jnp.sqrt(alphas_bar[i - 1])
        return (i - 1, rng, Ybar_im1), Yi,Y0s

    def reverse(YN, rng):
        Yi = YN
        for i in reversed(range(1,100)):
            carry = (i, rng, Yi)
            (i, rng, Yi),Yi_current,Y0s = reverse_once(carry)
            if args.save_video:
                # Salva Yi
                Yi_list.append(np.array(Yi_current))  # (H, n, Nu)
                Y0s_list.append(np.array(Y0s))  # (Nsample, H, n, Nu)

                # Rollout Yi
                _, traj_denoised = rollout_us(state_init, Yi_current)
                trajectories_denoised.append(np.array(traj_denoised[..., :2]))  # (T+1, n, 2)

                # Rollout Y0s
                _,traj_samples = jax.vmap(rollout_us, in_axes=(None, 0))(state_init, Y0s)
                trajectories_samples.append(np.array(traj_samples[..., :2]))  # (Nsample, T+1, n, 2)
        if args.save_video:
            return Yi, Yi_list, Y0s_list, trajectories_denoised, trajectories_samples
        else:
            return Yi
    if args.save_video:
        rng_exp, rng = jax.random.split(rng)
        U_0, Yi_list, Y0s_list, trajectories_denoised, trajectories_samples = reverse(YN, rng_exp)
        # Salva tutto
        np.savez("results/multicar_iterative/global_Yi_list.npz", 
                Yi_list=Yi_list, 
                Y0s_list=Y0s_list, 
                trajectories_denoised=trajectories_denoised, 
                trajectories_samples=trajectories_samples)
    else:
        rng_exp, rng = jax.random.split(rng)
        U_0 = reverse(YN, rng_exp)
    state_init_eval = reset_env_jit(jax.random.PRNGKey(args.seed + 1024))
    rewss_eval, _ = rollout_us(state_init_eval, U_0)
    rews, pipeline_states = rollout_us(state_init_eval, U_0)
   
    reward_per_robot = rewss_eval.mean(axis=0)
    reward_array_str = "[" + ", ".join(f"{r:.4f}" for r in reward_per_robot) + "]"
    print(f"global robots average rewards: {reward_array_str}")
    q_states = jnp.concatenate([state_init_eval.pipeline_state[None], pipeline_states[:-1]], axis=0)
    r_terms_all = []  # accumula r_terms: shape (n, 6)
    for t in range(U_0.shape[0]):
        q_t = q_states[t]
        u_t = jnp.clip(U_0[t], -1.0, 1.0)
        _, r_terms = env.get_rewards(q_t, u_t)   # shape (n, 6)
        r_terms_all.append(r_terms)
        



   


    return U_0

# Local iterative diffusion optimization
def run_diffusion_local(args: Args, U_init: jnp.ndarray,env, rollout_us, reset_env_jit):
    """
    Second phase of D4ORM: local iterative reverse diffusion optimization.
    Based on Algorithm 2 (Iterative Denoising) from the D4ORM paper.
    """
    rng = jax.random.PRNGKey(seed=args.seed + 123)  
    rewards_per_iter = []
    
    H = args.Hsample
    Nu = env.action_size
    n = env.num_robots

    U = U_init.copy()

    # Local diffusion parameters
    L = 10  # window length
    K = 5  # number of local iterations


    # Local diffusion schedule (more noisy)
    betas_local = jnp.linspace(0.01, 0.2, L)
    alphas_local = 1.0 - betas_local
    alphas_bar_local = jnp.cumprod(alphas_local)
    sigmas_local = jnp.sqrt(1 - alphas_bar_local) 

    lambda_goal = jnp.zeros((n * 2))

    

    def reverse_once_local_ECD(U_w, rng_w, lambda_goal, residual_fn, lagrangian):
            Nsample = args.Nsample
            N_inner = 30  # number of ECD iterations
            U_curr = U_w
            lambda_curr = lambda_goal
            delta_t = 0 
            for i in range(N_inner):

                rng_w, rng_step = jax.random.split(rng_w)
                # Noise annealing: exponentially decaying, then set to 0 in last 2 steps
                sigma_k = args.initial_sigma * jnp.exp(-args.noise_decay * i)
                sigma_k = jnp.where(i >= N_inner - 2, 0.0, sigma_k)
                mu_k = args.mu

                # Sample noisy control trajectories generated with gaussian noise
                eps_u = jax.random.normal(rng_step, (Nsample, L, n, Nu))
                noise = eps_u * sigma_k
                Y0s = U_curr + noise
                Y0s = jnp.clip(Y0s, -1.0, 1.0)
                
                U_fulls = jnp.repeat(U[None, ...], Nsample, axis=0)
                U_fulls = U_fulls.at[:, t_start:t_start + L, :, :].set(Y0s)
                Y0s_windows = Y0s[:, t_start:t_start + L, :, :]
                t1 = time.time()
                state_init = reset_env_jit(rng_w)
                rewss, pipeline_states = jax.vmap(rollout_us, in_axes=(None, 0))(state_init, U_fulls)
                t2 = time.time()
                delta_t += t2-t1 
                pipeline_states_window = pipeline_states[:, t_start:t_start+L]

                # Compute Lagrangian values for each sample
               # L_vals = lagrangian(Y0s, pipeline_states, lambda_curr, args.mu)
                L_cost,L_constraint,L_vals,control_cost,barrier_cost,goal_cost,h_flat,_ = lagrangian(Y0s_windows,Y0s, pipeline_states, pipeline_states_window, lambda_curr, args.mu)

                # Estimate gradient using score function estimator
                grad = jnp.einsum("s,slij->lij", L_vals - L_vals.mean(), noise)
                grad = grad / (Nsample * sigma_k ** 2 + 1e-8) # direzione del gradiente

                # === Clipping gradient
                # grad_norm = jnp.linalg.norm(grad)
                # grad = jnp.where(grad_norm > 1.0, grad * (1.0 / grad_norm), grad)

                # Update control using gradient descent and clip to [-1, 1]
                U_next = U_curr - args.alpha * grad
                U_next = jnp.clip(U_next, -1.0, 1.0)

                # Update Lagrange multipliers based on average residual
                h_goal = residual_fn(pipeline_states)
                h_mean = jnp.mean(h_goal, axis=0).reshape(-1)
                lambda_next = lambda_curr + args.alpha * mu_k * h_mean

                
                U_curr = U_next
                lambda_curr = lambda_next
            # print(f"Initial reverse diffusion time: {delta_t:.3f} s")
            return U_curr, lambda_curr

    if args.ECD:
        state_init_for_goal = reset_env_jit(jax.random.PRNGKey(args.seed + 777))
        residual_fn = make_residual_fn(state_init_for_goal, env, args.Nsample)
        lagrangian = make_lagrangian_fn(state_init_for_goal, env, args.Nsample)
        print("ECD finale")

        final_ecd_iters = 8
        for i in range(final_ecd_iters):
            for t_start in range(0, H - L + 1, L // 2):
                t_end = t_start + L
                U_window = U[t_start:t_end]
                rng, rng_step = jax.random.split(rng)
                U_opt_local, lambda_goal = reverse_once_local_ECD(U_window, rng_step, lambda_goal,residual_fn, lagrangian)
                U = U.at[t_start:t_end].set(U_opt_local)

            state_init_eval = reset_env_jit(jax.random.PRNGKey(args.seed + 999))
            rewss_eval, _ = rollout_us(state_init_eval, U)
            reward_per_robot = rewss_eval.mean(axis=0)
            rewards_per_iter.append(np.array(reward_per_robot))
            reward_array_str = "[" + ", ".join(f"{r:.4f}" for r in reward_per_robot) + "]"
            print(f"[Iteration {i}] robots average rewards: {reward_array_str}")
    else:
            for k in range(K):
                for t_start in range(0, H - L + 1, L // 2):  # sliding overlapping windows
                    t_end = t_start + L
                    U_window = U[t_start:t_end]
                    # sigma_local = sigmas_local[-1] # per ostacoli

                    rng, rng_step = jax.random.split(rng)

                    # Local reverse diffusion inside the window
                    def reverse_once_local(U_w, rng_w):
                        for j in reversed(range(1,L)):
                            eps_u = jax.random.normal(rng_w, (args.Nsample, L, n, Nu))
                            sigma_local = sigmas_local[j]
                            Y0s = eps_u * sigma_local + U_w
                            Y0s = jnp.clip(Y0s, -1.0, 1.0)

                            # Insert modified window into full control sequences
                            U_fulls = jnp.repeat(U[None, ...], args.Nsample, axis=0)
                            U_fulls = U_fulls.at[:, t_start:t_end, :, :].set(Y0s)
                            #  Evaluate new rollouts
                            state_init = reset_env_jit(rng_step)
                            rewss, _ = jax.vmap(rollout_us, in_axes=(None, 0))(state_init, U_fulls)
                            rews = rewss.mean(axis=(1, 2))

                            # Compute weighted average of samples
                            logp0 = (rews - rews.mean()) / (rews.std() + 1e-6) / args.temp_sample
                            weights = jax.nn.softmax(logp0)
                            U_opt = jnp.einsum("s,slij->lij", weights, Y0s)
                            # Final evaluation
                            U_w = jnp.sqrt(alphas_bar_local[j - 1]) * U_opt
                
                        return U_opt

                    U_opt_local = reverse_once_local(U_window, rng_step)
                    U = U.at[t_start:t_end].set(U_opt_local)
                
                state_init_eval = reset_env_jit(jax.random.PRNGKey(args.seed + 1024))
                rewss_eval, _ = rollout_us(state_init_eval, U)

                reward_per_robot = rewss_eval.mean(axis=0)
                rewards_per_iter.append(np.array(reward_per_robot))
                reward_array_str = "[" + ", ".join(f"{r:.4f}" for r in reward_per_robot) + "]"
                print(f"[Iteration {k}] robots average rewards: {reward_array_str}")

    return U,rewards_per_iter

def main():
    args = tyro.cli(Args)

    total_start = time.time()
   

    print("STEP 1: Initial Reverse Diffusion")
    env = MultiCar2d(n=args.n_robots, formation_shift=args.formation_shift,ECD=args.ECD, obstacles_enabled=args.obstacles_enabled)

    step_env_jit = jax.jit(env.step)
    reset_env_jit = jax.jit(env.reset)
    rollout_us = jax.jit(functools.partial(rollout_multi_us, step_env_jit))

    t1 = time.time()
    U_init = run_diffusion_once(args,env, rollout_us, reset_env_jit)
    t2 = time.time()
    print(f"Initial reverse diffusion time: {t2 - t1:.3f} s")

    print("STEP 2: Iterative Local Optimization")
    t3 = time.time()
    U_opt, rewards_per_iter = run_diffusion_local(args, U_init, env, rollout_us, reset_env_jit)
    t4 = time.time()
    print(f"Local optimization time: {t4 - t3:.3f} s")

    total_end = time.time()
    print(f"TOTAL planning time: {total_end - total_start:.3f} s")

    total_time = total_end - total_start
    freq = 1 / total_time
    print(f"Estimated control frequency: {freq:.2f} Hz")

    print("STEP 3: Rollout with optimized controls")


    state_init = reset_env_jit(jax.random.PRNGKey(args.seed + 1024))
    _, traj = rollout_us(state_init, U_opt)
    traj = jnp.concatenate([state_init.pipeline_state[None], traj], axis=0)
    traj = jnp.transpose(traj, (1, 0, 2))

    #print(f"Total runtime: {time.time() - start_time:.2f} s")

    print("Check collisions during rollout:")
    for t in range(traj.shape[1]):
        if check_inter_robot_collisions(traj[:, t, :], env.Ra):
            print(f"Collision detected at timestep {t}")

    # Compute final error
    x_goal = env.xg[:, :2]
    x_final = traj[:, -1, :2]
    errors = jnp.linalg.norm(x_final - x_goal, axis=1)

    print("\nErrore finale medio per robot:")
    for i, err in enumerate(errors):
        print(f"Robot {i}: {err:.3f} m")
    print(f"Mean distance to goal: {errors.mean():.4f} m\n")

    # interploate trajectory
    def interpolate_trajectory_jax(xs: jnp.ndarray, dt_original: float = 0.1, dt_interp: float = 0.01):
        n, T, d = xs.shape
        t_max = (T - 1) * dt_original
        t_interp = jnp.arange(0.0, t_max + dt_interp, dt_interp)

        t_original = jnp.arange(0.0, T * dt_original, dt_original)

        def interp_single_robot(traj):
            def interpolate_one(ti):
                idx = jnp.floor(ti / dt_original).astype(int)
                idx = jnp.clip(idx, 0, T - 2)
                t0 = t_original[idx]
                t1 = t_original[idx + 1]
                x0 = traj[idx]
                x1 = traj[idx + 1]
                alpha = (ti - t0) / (t1 - t0)
                return (1 - alpha) * x0 + alpha * x1

            return jax.vmap(interpolate_one)(t_interp)

        xs_interp = jax.vmap(interp_single_robot)(xs)
        return xs_interp, t_interp


    def plot_and_save_reward_terms(r_terms_all, path, prefix="reward", robot_ids=None, component_names=None):
        """
        Plotta e salva le componenti della reward per ciascun robot nel tempo.

        Args:
            r_terms_all: array shape (n, T+1, 6)
            path: cartella in cui salvare i plot
            prefix: prefisso nei nomi dei file
            robot_ids: lista opzionale di robot da plottare (default: tutti)
            component_names: nomi delle componenti della reward (default: autogenerati)
        """
        os.makedirs(path, exist_ok=True)
        n, T_plus_1, num_terms = r_terms_all.shape
        T = T_plus_1 - 1

        if robot_ids is None:
            robot_ids = list(range(n))
        if component_names is None:
            component_names = [f"term{i}" for i in range(num_terms)]

        for j in range(num_terms):
            plt.figure(figsize=(6, 4))
            for k in robot_ids:
                plt.plot(r_terms_all[k, :, j], label=f"Robot {k}")
            plt.xlabel("Tempo")
            plt.ylabel(component_names[j])
            plt.title(f"{component_names[j]} vs tempo")
            plt.grid(True)
            plt.legend()
            filename = f"{prefix}_{component_names[j]}.pdf"
            plt.tight_layout()
            plt.savefig(os.path.join(path, filename))
            plt.close()
            print(f"Salvato: {os.path.join(path, filename)}")

    if not args.not_render:
        path = "results/multicar_iterative"
        os.makedirs(path, exist_ok=True)


        state_init = reset_env_jit(jax.random.PRNGKey(args.seed + 1024))
        xs = jnp.array([state_init.pipeline_state])
        r_terms_all_global = jnp.array([jnp.zeros((args.n_robots, 6))])  # shape (1, n, 6)
        state = state_init
        r_terms_all=[]
        for t in range(U_opt.shape[0]):
            state = step_env_jit(state, U_opt[t])
            xs = jnp.concatenate([xs, state.pipeline_state[None]], axis=0)
            # calcolo reward divise
            _, r_terms = env.get_rewards(state.pipeline_state, U_opt[t])  # r_terms: (n, 6)
            r_terms_all_global = jnp.concatenate([r_terms_all_global, r_terms[None]], axis=0)
            r_terms_all = jnp.stack(r_terms_all_global, axis=1)  # shape (n, T, 6)

        xs = jnp.transpose(xs, (1, 0, 2))
    
        r_total_from_terms = r_terms_all[:, :, -1]         # shape (n, T)
        r_total_mean = r_total_from_terms.mean(axis=-1)     # shape (n,)
        print("From get_rewards:", r_total_mean)
        r_terms_all_global = jnp.transpose(r_terms_all_global, (1, 0, 2))  # (n, T+1, 6)
        
         # === Rollout iniziale (U_init) ===
        state_init = reset_env_jit(jax.random.PRNGKey(args.seed + 1024))
        x_init = jnp.array([state_init.pipeline_state])
        state = state_init
        r_terms_all_local = jnp.array([jnp.zeros((args.n_robots, 6))])  # shape (1, n, 6)
        for t in range(U_init.shape[0]):
        
            u_t = jnp.clip(U_init[t], -1.0, 1.0)
       
            state = step_env_jit(state, U_init[t])
            x_init = jnp.concatenate([x_init, state.pipeline_state[None]], axis=0)
            _, r_terms = env.get_rewards(state.pipeline_state, u_t)  # r_terms: (n, 6)
            r_terms_all_local = jnp.concatenate([r_terms_all_local, r_terms[None]], axis=0)
        x_init = jnp.transpose(x_init, (1, 0, 2))

        r_terms_all_local = jnp.transpose(r_terms_all_local, (1, 0, 2))  # (n, T+1, 6)
       
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.set_aspect('equal', adjustable='datalim')
        # Traiettoria iniziale in grigio tratteggiato
        cmap = plt.get_cmap('tab20', x_init.shape[0])
        for i in range(x_init.shape[0]):
            ax.plot(x_init[i, :, 0], x_init[i, :, 1], '--', alpha=0.3, color=cmap(i),label = f"Robot {i} global")

        env.render(ax, xs, goals=env.xg)
        
        ecd_tag = "ecd" if args.ECD else "d4orm"
        formation_tag = "form" if args.formation_shift else ""
        plt.title(f"Optimized final trajector {ecd_tag}_{formation_tag}")
        plt.tight_layout()
        plt.savefig(os.path.join(path, f"local_diffusion_{ecd_tag}_{formation_tag}.pdf"))
        print(f"Figura salvata in {path}/local_diffusion.pdf")
        output_dir_local = os.path.join(path, "reward_local")
        output_dir_global = os.path.join(path, "reward_global")
        component_names = ["r_goal", "r_safe", "r_form", "r_control", "r_obs", "r_total"]
        plot_and_save_reward_terms(r_terms_all_local, path=output_dir_local, prefix=f"reward_plot_{ecd_tag}_{formation_tag}",component_names=component_names)
        plot_and_save_reward_terms(r_terms_all_global, path=output_dir_global, prefix=f"globalreward_plot_{ecd_tag}_{formation_tag}",component_names=component_names)

        # Save trajectory data and optimized control
        
        np.savez(f"results/multicar_iterative/optimized_data_{ecd_tag}_{formation_tag}.npz", U_opt=np.array(U_opt), traj=np.array(traj), goals=np.array(env.xg), rewards=np.array(rewards_per_iter))

        #Create video

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
        ax.set_aspect('equal', adjustable='box')

        lines = [ax.plot([], [], 'o', label=f"Robot {i}")[0] for i in range(n)]
        goals = env.xg if hasattr(env, "xg") else None

        # Calcolate bounding box 
        x_all = xs_np[:, :, 0].flatten()
        y_all = xs_np[:, :, 1].flatten()

        x_min, x_max = x_all.min(), x_all.max()
        y_min, y_max = y_all.min(), y_all.max()

        margin = 2  

        ax.set_xlim(float(x_min - margin), float(x_max + margin))
        ax.set_ylim(float(y_min - margin), float(y_max + margin))

        ax.set_title("Robot tracking")
       
        if args.formation_shift:
            c0 = env.x0[:, :2].mean(axis=0)
            circle0 = plt.Circle((c0[0], c0[1]), env.radius, color='gray', linestyle='--', fill=False)
            ax.add_patch(circle0)
            cg = env.xg[:, :2].mean(axis=0)
            circleg = plt.Circle((cg[0], cg[1]), env.radius, color='black', linestyle='--', fill=False)
            ax.add_patch(circleg)

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

        orientation_arrows = []

        def update(frame):
            for i in range(n):
                x_trail = xs_np[i, :frame + 1, 0]
                y_trail = xs_np[i, :frame + 1, 1]
                lines[i].set_data(x_trail, y_trail)

                x_curr = xs_np[i, frame, 0]
                y_curr = xs_np[i, frame, 1]
                points[i].set_data([x_curr], [y_curr])
                # Rimuovi freccia precedente
                for arr in orientation_arrows:
                    arr.remove()
                orientation_arrows.clear()

                for i in range(n):
                    x_trail = xs_np[i, :frame + 1, 0]
                    y_trail = xs_np[i, :frame + 1, 1]
                    lines[i].set_data(x_trail, y_trail)

                    x_curr = xs_np[i, frame, 0]
                    y_curr = xs_np[i, frame, 1]
                    theta_curr = xs_np[i, frame, 2]  # Assumiamo che theta sia in posizione 2

                    points[i].set_data([x_curr], [y_curr])

                    dx = 0.3 * np.cos(theta_curr)
                    dy = 0.3 * np.sin(theta_curr)
                    arrow = ax.arrow(x_curr, y_curr, dx, dy, head_width=0.1, head_length=0.15, fc=cmap(i), ec=cmap(i))
                    orientation_arrows.append(arrow)

            return lines + points+ orientation_arrows

        ani = animation.FuncAnimation(
            fig, update, frames=T, init_func=init, blit=False, interval=100
        )
        labels = [f"Robot {i}" for i in range(n)]
        for x_c, y_c, w, h in env.static_obstacles:
            rect = plt.Rectangle((x_c - w / 2, y_c - h / 2), w, h,
                                linewidth=1, edgecolor='red', facecolor='red', alpha=0.5)
            ax.add_patch(rect)
        ax.legend(points, labels, loc='upper right')


        video_path = os.path.join(path, f"local_diffusion_{ecd_tag}_{formation_tag}.mp4")
        ani.save(video_path, fps=10, dpi=150)
        print("Video saved in:", video_path)



if __name__ == "__main__":
    main()