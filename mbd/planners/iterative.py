import functools
import jax
from jax import numpy as jnp

import os
from mbd.utils import rollout_multi_us , make_lagrangian_fn, make_residual_fn
from mbd.envs.multi_car import check_inter_robot_collisions, Args, check_collision_static
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mbd.envs import MultiCar2d

import tyro
import numpy as np
import time
from mbd.butterworth import butterworth_filter_numpy,ar1_noise_numpy
from mbd.butterworth import get_butterworth_coeffs

    # Single-pass reverse diffusion to initialize U
def ar1_noise(key, shape, rho=0.99999):
    """
    shape: (Nsample, H, n, Nu)
    output: (N, T, n, Nu)
    """
    N, T, n, Nu = shape
    key_init, key_scan = jax.random.split(key)
    eps0 = jax.random.normal(key_init, (N, n, Nu))  # (N, n, Nu)

    def ar1_step(eps_prev, key_t):
        noise_t = jax.random.normal(key_t, (N, n, Nu))
        eps_t = rho * eps_prev + jnp.sqrt(1 - rho**2) * noise_t
        return eps_t, eps_t

    keys_scan = jax.random.split(key_scan, T - 1)
    _, eps_seq = jax.lax.scan(ar1_step, eps0, keys_scan)  # (T-1, N, n, Nu)
    eps_seq = jnp.transpose(eps_seq, (1, 0, 2, 3))        # (N, T-1, n, Nu)

    eps_full = jnp.concatenate([eps0[:, None], eps_seq], axis=1)  # (N, T, n, Nu)
    return eps_full

def cosine_beta_schedule(T, s=0.008):
    t = jnp.arange(T + 1, dtype=jnp.float32)
    f_t = jnp.cos(((t / T + s) / (1 + s)) * jnp.pi / 2) ** 2
    alphas_bar = f_t / f_t[0]
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    betas = 1 - alphas
    return jnp.clip(betas, 1e-5, 0.999)

def cosine_beta_schedule_scaled(T, beta0, betaT, s=0.008):
    """
    Cosine schedule con scaling per rendere beta0 e betaT coerenti con valori desiderati.
    """
    t = jnp.arange(T + 1, dtype=jnp.float32)
    f_t = jnp.cos(((t / T + s) / (1 + s)) * jnp.pi / 2) ** 2
    alphas_bar = f_t / f_t[0]
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    betas = 1 - alphas

    beta_min, beta_max = betas.min(), betas.max()
    betas_scaled = (betas - beta_min) / (beta_max - beta_min)  # ∈ [0,1]
    betas_scaled = betas_scaled * (betaT - beta0) + beta0      # ∈ [beta0, betaT]

    return betas_scaled



def run_diffusion_once(args: Args,env, rollout_us, reset_env_jit):
    """
    First phase of D4ORM: initial global reverse diffusion
    inspired by Algorithm 1 (Model-Based Diffusion) from the D4ORM paper.

    """
    rng = jax.random.PRNGKey(seed=args.seed)
 
    Nx = env.observation_size
    Nu = env.action_size
    n = env.num_robots



    rng, rng_reset = jax.random.split(rng)
    state_init = reset_env_jit(rng_reset)
    if args.save_video:
        Yi_list = []
        Y0s_list = []
        trajectories_denoised = []
        trajectories_samples = []

    # Diffusion noise schedule
    betas = jnp.linspace(args.beta0, args.betaT, args.Ndiffuse)
    # betas = cosine_beta_schedule(args.Ndiffuse)
    #betas = cosine_beta_schedule_scaled(args.Ndiffuse, args.beta0, args.betaT)


    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)
    sigmas = jnp.sqrt(1 - alphas_bar)

    #  Start from zero control
    YN = jnp.zeros([args.Hsample, n, Nu])
    
    # Single diffusion step
    #@jax.jit
    def reverse_once(carry):
        i, rng, Ybar_i = carry
        Yi = Ybar_i * jnp.sqrt(alphas_bar[i])

        # Sample noisy controls
        rng, rng_eps = jax.random.split(rng)
        eps_u = jax.random.normal(rng_eps, (args.Nsample, args.Hsample, n, Nu))
        #eps_u = ar1_noise(rng_eps, (args.Nsample, args.Hsample, n, Nu), rho=0.9)
       
        

        # eps_u_np = np.array(eps_u)
        # b, a = get_butterworth_coeffs(order=4, fc=2.0, fs=1/env.dt)  # fc personalizzata
        # eps_u_filt_np = butterworth_filter_numpy(eps_u_np, b, a)
        # eps_u = jnp.array(eps_u_filt_np)  # torna in JAX


        Y0s = eps_u * sigmas[i] + Ybar_i
        if env.obstacles_enabled == False:
            Y0s = jnp.clip(Y0s, -1.0, 1.0)
        #jax.debug.print("Y0s min: {0:.2f}, max: {1:.2f}", jnp.min(Y0s), jnp.max(Y0s))
        rewss, _ = jax.vmap(rollout_us, in_axes=(None, 0))(state_init, Y0s)
        rews = rewss.mean(axis=(1, 2))
        # jax.debug.print("→  reward mean: {:.3f}, max: {:.3f}", rews.mean(), rews.max())

        rew_std = rews.std()
        rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std)
        logp0 = (rews - rews.mean()) / rew_std / args.temp_sample
        # print(f"[Reverse step {i}] reward mean: {rews.mean():.3f}, std: {rews.std():.3f}, max: {rews.max():.3f}, min: {rews.min():.3f}")
        # print(f"logp0 range: [{logp0.min():.3f}, {logp0.max():.3f}]")

        # if i % 10 == 0:
        #     # Debug: range dei reward
        #     print("Reward min/max:", rews.min(), rews.max())
        #     # Debug: varianza dei sample (quanto i controlli sono diversi tra loro)
        #     print("Varianza Y0s:", Y0s.var())
        #     # Debug: media campioni (vedi se collassano a zero)
        #     print("Mean Y0s:", Y0s.mean())
        # Weighted average of samples  (Monte carlo estimate)
        weights = jax.nn.softmax(logp0)
        # print(f"softmax weights max: {weights.max():.3f}, min: {weights.min():.3f}")

        Ybar = jnp.einsum("s,shij->hij", weights, Y0s)
        # best = jnp.argmax(logp0)
        # Ybar = Y0s[best]
        # Reverse diffusion step
        score = 1 / (1.0 - alphas_bar[i]) * (-Yi + jnp.sqrt(alphas_bar[i]) * Ybar)
        Yim1 = 1 / jnp.sqrt(alphas[i]) * (Yi + (1.0 - alphas_bar[i]) * score)
        Ybar_im1 = Yim1 / jnp.sqrt(alphas_bar[i - 1])
        return (i - 1, rng, Ybar_im1), Yi,Y0s

    def reverse(YN, rng):
        Yi = YN
        for i in reversed(range(1,args.Ndiffuse)):
            carry = (i, rng, Yi)
            (i, rng, Yi),Yi_current,Y0s = reverse_once(carry)
            if args.save_video and i % 10  == 0: 
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
        #u_t = jnp.clip(U_0[t], -1.0, 1.0)
        u_t = U_0[t]
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
    if args.save_video:
        trajectories_denoised_local = []
        trajectories_samples_local = []

    
    H = args.Hsample
    Nu = env.action_size
    n = env.num_robots

    U = U_init.copy()

    # Local diffusion parameters
    L = 10  # window length
    K = 10  # number of local iterations


    # Local diffusion schedule (more noisy)
    # betas_local = jnp.linspace(0.01, 0.2, L)
    # alphas_local = 1.0 - betas_local
    # alphas_bar_local = jnp.cumprod(alphas_local)
    # sigmas_local = jnp.sqrt(1 - alphas_bar_local) 
    betas = jnp.linspace(args.beta0, args.betaT, 10)
    #betas = exponential_beta_schedule(n_diffusion_steps=100, beta_start=args.beta0, beta_end=args.betaT)

    alphas = 1.0 - betas
    alphas_bar_local = jnp.cumprod(alphas)
    sigmas_local = jnp.sqrt(1 - alphas_bar_local)


    lambda_goal = jnp.zeros((n * 2))

    

    def reverse_once_local_ECD(U_w, rng_w, lambda_goal, residual_fn, lagrangian, U_full, t_start, reset_env_jit, rollout_us):

            Nsample = args.Nsample
            N_inner = 30  # number of ECD iterations
            U_curr = U_w
            lambda_curr = lambda_goal
            delta_t = 0 
            # U_soft_single = U_full  # shape (L, n, Nu)
        
            # state_soft = env.reset(jax.random.PRNGKey(args.seed ))
            # rew_soft, pipeline_soft = rollout_us(state_soft, U_soft_single)
            # U_soft_batch = jnp.repeat(U_soft_single[None, ...], args.Nsample, axis=0)  # shape (Nsample, L, n, Nu)
            # state_soft = reset_env_jit(jax.random.PRNGKey(args.seed))
            # #rew_soft, pipeline_soft = jax.vmap(rollout_us, in_axes=(None, 0))(state_soft, U_soft_batch)

            
            # # Usa una nuova lagrangiana con batch size = 1
            # lagrangian_fn_1 = make_lagrangian_fn(state_soft, env, Nsample=1)

            # L_soft, _, L_tot_soft, *_ = lagrangian_fn_1(
            #     U_soft_single[None, ...], U_soft_single[None, ...], 
            #     pipeline_soft[None, ...], pipeline_soft[None, ...], 
            #     lambda_curr, args.mu
            # )

            # baseline =  L_tot_soft[0] 
            # print(f"   baseline = {baseline:.4f}")
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
                # nota : qui il clip va restato 
                Y0s = jnp.clip(Y0s, -1.0, 1.0)
                
                U_fulls = jnp.repeat(U_full[None, ...], Nsample, axis=0)
                U_fulls = U_fulls.at[:, t_start:t_start + L, :, :].set(Y0s)
                #Y0s_windows = Y0s[:, t_start:t_start + L, :, :]
                t1 = time.time()
                state_init = reset_env_jit(rng_w)
                state_init = reset_env_jit(rng_w)

                rewss, pipeline_states = jax.vmap(rollout_us, in_axes=(None, 0))(state_init, U_fulls)
                t2 = time.time()
                delta_t += t2-t1 
                pipeline_states_window = pipeline_states[:, t_start:t_start+L]

                # Compute Lagrangian values for each sample
                L_cost,L_constraint,L_vals,control_cost,barrier_cost,goal_cost,h_flat,obstacle_cost_global,orient_cost_global,reverse_penalty_global = lagrangian(Y0s,U_fulls, pipeline_states, pipeline_states_window, lambda_curr, args.mu)

                # Estimate gradient using score function estimator
                grad = jnp.einsum("s,slij->lij", L_vals - L_vals.mean(), noise)
                grad = grad / (Nsample * sigma_k ** 2 + 1e-8) # direzione del gradiente

                
                rng_key, rng_noise = jax.random.split(rng_w)
                xi = jax.random.normal(rng_noise, U_curr.shape)
                sigma_langevin = sigma_k * jnp.sqrt(2 * args.alpha)
                noise_term = sigma_langevin * xi

                U_next = U_curr - args.alpha * grad + noise_term
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
                U_opt_local, lambda_goal = reverse_once_local_ECD(U_window, rng, lambda_goal, residual_fn, lagrangian, U, t_start, reset_env_jit, rollout_us)
                U = U.at[t_start:t_end].set(U_opt_local)

            state_init_eval = reset_env_jit(jax.random.PRNGKey(args.seed ))
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
                            if env.obstacles_enabled == False:
                                 Y0s = jnp.clip(Y0s, -1.0, 1.0)
                            

                            # Insert modified window into full control sequences
                            U_fulls = jnp.repeat(U[None, ...], args.Nsample, axis=0)
                            U_fulls = U_fulls.at[:, t_start:t_end, :, :].set(Y0s)
                            #  Evaluate new rollouts
                            state_init = reset_env_jit(rng_step)
                            rewss, traj_samples = jax.vmap(rollout_us, in_axes=(None, 0))(state_init, U_fulls)
                            rews = rewss.mean(axis=(1, 2))

                            # Compute weighted average of samples
                            logp0 = (rews - rews.mean()) / (rews.std() + 1e-6) / args.temp_sample
                            weights = jax.nn.softmax(logp0)
                            U_opt = jnp.einsum("s,slij->lij", weights, Y0s)
                            # best = jnp.argmax(logp0)
                            # U_opt = Y0s[best]
                            # Final evaluation
                            U_new = jnp.sqrt(alphas_bar_local[j - 1]) * U_opt
                            # if args.save_video:
                            #     # Rollout per ottenere (x,y) delle traiettorie campionate
                            #     _, traj_denoised = rollout_us(state_init, U_new)

                            #     trajectories_samples_local.append(np.array(traj_samples[..., :2]))       # shape: (Nsample, T+1, n, 2)
                            #     trajectories_denoised_local.append(np.array(traj_denoised[..., :2]))     # shape: (T+1, n, 2)

                
                        return U_new

                    U_opt_local = reverse_once_local(U_window, rng_step)
                    U = U.at[t_start:t_end].set(U_opt_local)
                
                state_init_eval = reset_env_jit(jax.random.PRNGKey(args.seed + 1024))
                
                rewss_eval, _ = rollout_us(state_init_eval, U)

                reward_per_robot = rewss_eval.mean(axis=0)
                rewards_per_iter.append(np.array(reward_per_robot))
                reward_array_str = "[" + ", ".join(f"{r:.4f}" for r in reward_per_robot) + "]"
                print(f"[Iteration {k}] robots average rewards: {reward_array_str}")
    if args.save_video:np.savez("results/multicar_iterative/local_Yi_list.npz", 
             trajectories_denoised=trajectories_denoised_local, 
             trajectories_samples=trajectories_samples_local)

    return U,rewards_per_iter

def main():
    args = tyro.cli(Args)

    total_start = time.time()
   

    print("STEP 1: Initial Reverse Diffusion")
    env = MultiCar2d(n=args.n_robots, formation_shift=args.formation_shift,ECD=args.ECD, obstacles_enabled=args.obstacles_enabled,penalize_backward=args.penalize_backward)

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
        if check_collision_static(traj[:, t, :], env.static_obstacles):
            print(f"Collision with static obstacles detected at timestep {t}")
   



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
            #    print(f"Salvato: {os.path.join(path, filename)}")

    if not args.not_render:
        path = "results/multicar_iterative"
        os.makedirs(path, exist_ok=True)


        state_init = reset_env_jit(jax.random.PRNGKey(args.seed + 1024))
        x_init = jnp.array([state_init.pipeline_state])
        r_terms_all_global = jnp.array([jnp.zeros((args.n_robots, 7))])  # shape (1, n, 6)
        state = state_init
        r_terms_all=[]
        for t in range(U_init.shape[0]):
            u_t = jnp.clip(U_init[t], -1.0, 1.0)
            #u_t = U_init[t]
            state = step_env_jit(state, U_init[t])
            x_init = jnp.concatenate([x_init, state.pipeline_state[None]], axis=0)
            # calcolo reward divise
            _, r_terms = env.get_rewards(state.pipeline_state, u_t)  # r_terms: (n, 6)
            r_terms_all.append(r_terms)
            r_terms_all_global = jnp.concatenate([r_terms_all_global, r_terms[None]], axis=0)
        

        x_init = jnp.transpose(x_init, (1, 0, 2))
        r_terms_all = jnp.stack(r_terms_all, axis=1)  # shape (n, T, 6)
         # Save final trajectory plot
        r_total_from_terms = r_terms_all[:, :, -1]         # shape (n, T)
        r_total_mean = r_total_from_terms.mean(axis=-1) 
        print("From get_rewards:", r_total_mean)
        
        r_terms_all_global = jnp.transpose(r_terms_all_global, (1, 0, 2))  # (n, T+1, 6)
        
         # === Rollout iniziale (U_init) ===
        state_init = reset_env_jit(jax.random.PRNGKey(args.seed + 1024))
        xs = jnp.array([state_init.pipeline_state])
        state = state_init
        r_terms_all_local = jnp.array([jnp.zeros((args.n_robots, 7))])  # shape (1, n, 6)
        for t in range(U_opt.shape[0]):
            #u_t = jnp.clip(U_opt[t], -1.0, 1.0)
            u_t = U_opt[t]      
            state = step_env_jit(state, U_opt[t])
            xs = jnp.concatenate([xs, state.pipeline_state[None]], axis=0)
            _, r_terms = env.get_rewards(state.pipeline_state, u_t)  # r_terms: (n, 6)
            r_terms_all_local = jnp.concatenate([r_terms_all_local, r_terms[None]], axis=0)
        xs = jnp.transpose(xs, (1, 0, 2))

        r_terms_all_local = jnp.transpose(r_terms_all_local, (1, 0, 2))  # (n, T+1, 6)
       
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.set_aspect('equal', adjustable='datalim')
        # Traiettoria iniziale in grigio tratteggiato
        cmap = plt.get_cmap('tab20', x_init.shape[0])
        print("Shape x_init:", x_init.shape)
        for i in range(x_init.shape[0]):
            ax.plot(x_init[i, :, 0], x_init[i, :, 1], '--', color=cmap(i),label = f"Robot {i} global")

        env.render(ax, xs, goals=env.xg,actions = U_opt)
        
        ecd_tag = "ecd" if args.ECD else "d4orm"
        formation_tag = "form" if args.formation_shift else ""
        plt.title(f"Optimized final trajector {ecd_tag}_{formation_tag}")
        plt.tight_layout()
        plt.savefig(os.path.join(path, f"local_diffusion_{ecd_tag}_{formation_tag}.pdf"))
        print(f"Figura salvata in {path}/local_diffusion.pdf")
        

        # === Seconda figura: solo env.render ===
        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5))
        ax2.set_aspect('equal', adjustable='datalim')
        env.render(ax2, xs, goals=env.xg,actions=U_opt)
        plt.title(f"Optimized final render only {ecd_tag}_{formation_tag}")
        plt.tight_layout()
        plt.savefig(os.path.join(path, f"traiettoria_finale{ecd_tag}_{formation_tag}.pdf"))
        print(f"Figura salvata in {path}/traiettoria_finale{ecd_tag}_{formation_tag}.pdf")


        output_dir_local = os.path.join(path, "reward_local")
        output_dir_global = os.path.join(path, "reward_global")
        component_names = ["r_goal", "r_safe", "r_form", "r_control", "r_obs","r_backward","r_total"]
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
        actions = jnp.transpose(U_opt, (1, 0, 2))  # (n, T, Nu)

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
            line, = ax.plot([], [], lw=2,color = color,zorder=1)  
            lines.append(line)


            #point, = ax.plot([], [], 'o', markersize=6,color = color)
            #points.append(point)

            if goals is not None:
                    gx, gy = goals[i, 0], goals[i, 1]
                    ax.plot(gx, gy, 's', color=color, markersize=6, markeredgewidth=2)

        orientation_arrows = []
        robot_circles = []
        def update(frame):
             # Rimuovi freccia precedente
            for arr in orientation_arrows:
                arr.remove()
            orientation_arrows.clear()
            # Rimuovi cerchi precedenti
            for c in robot_circles:
                c.remove()
            robot_circles.clear()
            for i in range(n):
                x_trail = xs_np[i, :frame + 1, 0]
                y_trail = xs_np[i, :frame + 1, 1]
                lines[i].set_data(x_trail, y_trail)

                x_curr = xs_np[i, frame, 0]
                y_curr = xs_np[i, frame, 1]
                theta_curr = xs_np[i, frame, 2]  

                #points[i].set_data([x_curr], [y_curr])
                
                circle = plt.Circle((x_curr, y_curr), 0.1, color=cmap(i), fill=False, linestyle='-', linewidth=1,zorder=3)
                ax.add_patch(circle)
                robot_circles.append(circle)

                dx = 0.3 * np.cos(theta_curr)
                dy = 0.3 * np.sin(theta_curr)

                v_curr = actions[i, frame, 1]
                color_arrow = 'green' if v_curr >= -0.05 else 'red'

                arrow = ax.arrow(x_curr, y_curr, dx, dy, head_width=0.1, head_length=0.15, fc=color_arrow, ec=color_arrow)

                orientation_arrows.append(arrow)                    

            return lines + points+ orientation_arrows

        ani = animation.FuncAnimation(
            fig, update, frames=T, init_func=init, blit=False, interval=100
        )
        labels = [f"Robot {i}" for i in range(n)]
        buffer_min = 0.2
        buffer_max = 0.5

        for x_c, y_c, w, h in env.static_obstacles:
            rect_outer = plt.Rectangle(
                (x_c - (w / 2 + buffer_max), y_c - (h / 2 + buffer_max)),
                w + 2 * buffer_max,
                h + 2 * buffer_max,
                linewidth=0,
                facecolor='yellow',
                alpha=0.1,
                zorder=1
            )
            ax.add_patch(rect_outer)

        for x_c, y_c, w, h in env.static_obstacles:
            rect_inner = plt.Rectangle(
                (x_c - (w / 2 + buffer_min), y_c - (h / 2 + buffer_min)),
                w + 2 * buffer_min,
                h + 2 * buffer_min,
                linewidth=0,
                facecolor='yellow',
                alpha=0.5,
                zorder=2
            )
            ax.add_patch(rect_inner)

        for x_c, y_c, w, h in env.static_obstacles:
            rect_real = plt.Rectangle(
                (x_c - w / 2, y_c - h / 2),
                w, h,
                linewidth=1,
                edgecolor='red',
                facecolor='red',
                zorder=3
            )
            ax.add_patch(rect_real)


        ax.legend(points, labels, loc='upper right')


        video_path = os.path.join(path, f"local_diffusion_{ecd_tag}_{formation_tag}.mp4")
        ani.save(video_path, fps=10, dpi=150)
        print("Video saved in:", video_path)



if __name__ == "__main__":
    main()