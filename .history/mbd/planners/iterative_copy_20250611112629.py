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

import pickle
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

    rng, rng_reset = jax.random.split(rng)
    state_init = reset_env_jit(rng_reset)

    # Diffusion noise schedule
    betas = jnp.linspace(args.beta0, args.betaT, 20)
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
        weights = jax.nn.softmax(logp0) # trasformazione in pesi 
        Ybar = jnp.einsum("s,shij->hij", weights, Y0s)
        # Reverse diffusion step
        score = 1 / (1.0 - alphas_bar[i]) * (-Yi + jnp.sqrt(alphas_bar[i]) * Ybar)
        Yim1 = 1 / jnp.sqrt(alphas[i]) * (Yi + (1.0 - alphas_bar[i]) * score)
        Ybar_im1 = Yim1 / jnp.sqrt(alphas_bar[i - 1])
        return (i - 1, rng, Ybar_im1), None

    def reverse(YN, rng):
        Yi = YN
        for i in reversed(range(1,20)):
            carry = (i, rng, Yi)
            (i, rng, Yi), _ = reverse_once(carry, None)
        return Yi  # U^(0)

    rng_exp, rng = jax.random.split(rng)
    U_0 = reverse(YN, rng_exp)
    state_init_eval = reset_env_jit(jax.random.PRNGKey(args.seed + 1024))
    rewss_eval, _ = rollout_us(state_init_eval, U_0)

    reward_per_robot = rewss_eval.mean(axis=0)
   
    reward_array_str = "[" + ", ".join(f"{r:.4f}" for r in reward_per_robot) + "]"
    print(f"[global robots average rewards: {reward_array_str}")
    return U_0

# Local iterative diffusion optimization
def run_diffusion_local(args: Args, U_init: jnp.ndarray,env, rollout_us, reset_env_jit):
    """
    Second phase of D4ORM: local iterative reverse diffusion optimization.
    Based on Algorithm 2 (Iterative Denoising) from the D4ORM paper.
    """
    rng = jax.random.PRNGKey(seed=args.seed + 123)  # different seed for local phase
    rewards_per_iter = []
    H = args.Hsample
    Nu = env.action_size
    n = env.num_robots

    U = U_init.copy()

    # Local diffusion parameters
    L = 10  # window length
    K = 5  # number of local iterations


    # Local diffusion schedule (more noisy)
    betas_local = jnp.linspace(0.01, 0.2, 10)
    alphas_local = 1.0 - betas_local
    alphas_bar_local = jnp.cumprod(alphas_local)
    sigmas_local = jnp.sqrt(1 - alphas_bar_local)

    sigma_local = sigmas_local[-1]

    
    lambda_goal = jnp.zeros((env.n * 2,))


    # Preparazione funzioni
    state_init_for_goal = reset_env_jit(jax.random.PRNGKey(args.seed + 777))
    residual_fn = make_residual_fn(state_init_for_goal, env, args.Nsample)
    lagrangian_fn = make_lagrangian_fn(state_init_for_goal, env, args.Nsample)
    R_window_all = []
    J_goal_all = []
    J_barrier_all = []
    J_control_all = []
    H_norm_all = []

      
    def reverse_once_local_hybrid(U_window, rng_w, U_full_template, t_start, residual_fn, lagrangian_fn, lambda_goal, args, env,sigma_k):
        Nsample = args.Nsample
        L, n, Nu = U_window.shape
        sigma_local =  0.2
   
        # Step 1: Sampling (diffusion-based denoising)
        eps_u = jax.random.normal(rng_w, (Nsample, L, n, Nu))
        Y0s = eps_u * sigma_local + U_window
        noise = eps_u*sigma_local
        noise_norm = jnp.linalg.norm(noise.reshape(Nsample, -1), axis=1).mean()
        Y0s = jnp.clip(Y0s, -1.0, 1.0)

        # Step 2: Costruisci rollouts completi modificando la finestra
        U_fulls = jnp.repeat(U_full_template[None, ...], Nsample, axis=0)
        U_fulls = U_fulls.at[:, t_start:t_start+L, :, :].set(Y0s)
      
        Y0s_window = Y0s[:, t_start:t_start+L, :, :]
        state_init = env.reset(jax.random.PRNGKey(args.seed +1024))  
        rewss, pipeline_states = jax.vmap(rollout_us, in_axes=(None, 0))(state_init, U_fulls)
        pipeline_states_window = pipeline_states[:, t_start:t_start+L]
        r_vals_window = rewss[:, t_start:t_start+L, :].mean(axis=(1,2))
        # Step 3: Calcola loss Lagrangiana come score
        L_cost,L_constraint,L_tot,control_cost,barrier_cost,goal_cost,h_flat,_ = lagrangian_fn(Y0s_window,Y0s, pipeline_states, pipeline_states_window, lambda_goal, args.mu)

        
        alpha = 1.0  
        r_std = (r_vals_window - r_vals_window.mean()) / (r_vals_window.std() + 1e-6)
        L_std = (L_cost - L_cost.mean()) / (L_cost.std() + 1e-6)
        
        score_vals = r_std - alpha * L_std
   
        logp0 = (score_vals - score_vals.mean()) / (score_vals.std() + 1e-6) / args.temp_sample
        weights = jax.nn.softmax(logp0)
        
   
        U_soft = jnp.einsum("s,slij->lij", weights, Y0s)

        # # Step 2: gradiente REINFORCE su L_constraint
        
        grad = jnp.einsum("s,slij->lij", (L_tot - L_tot.mean()), noise)
        grad = grad / (Nsample * sigma_local**2 + 1e-6)
        

    #     # === Rollout singolo su U_soft ===
       
    #     U_soft_batched = jnp.repeat(U_soft[None, ...], Nsample, axis=0)  
    #     U_full_soft = jnp.repeat(U_full_template[None, ...], Nsample, axis=0)
    #     U_full_soft = U_full_soft.at[:, t_start:t_start+L, :, :].set(U_soft_batched)

    #     # Rollout batch con traiettoria centrale
    #     state_init_eval = env.reset(jax.random.PRNGKey(args.seed + 1024))
    #     rewss_soft, pipeline_states_soft = jax.vmap(rollout_us, in_axes=(None, 0))(state_init_eval, U_full_soft)
    #     pipeline_states_window_soft = pipeline_states_soft[:, t_start+1:t_start+L+1, :, :]

    #     # === Lagrangiana su batch centrale (identico in tutti i sample) ===
    #     _, _, L_base, *_ = lagrangian_fn(
    #         U_soft_batched,               
    #         U_soft_batched,              
    #         pipeline_states_soft[:, 1:], 
    #         pipeline_states_window_soft,
    #         lambda_goal,
    #         args.mu
    #     )

    #     L_x = L_base  # shape: (Nsample,)


    #    # # REINFORCE-style gradient
    #     grad = jnp.einsum("s,slij->lij", (L_tot - L_x), noise)
    #     grad = grad / (Nsample * sigma_local**2 + 1e-6)

        U_next = U_soft - 0.01* grad
       
        
        h_goal = residual_fn(pipeline_states)  
        h_mean = jnp.mean(h_goal, axis=0).reshape(-1)
        
        lambda_goal = lambda_goal + args.alpha * args.mu * h_mean
                
        
        # print("L_cost mean/std:", L_cost.mean(), L_cost.std())
        # print("r_vals_window mean/std:", r_vals_window.mean(), r_vals_window.std())


        return U_next, lambda_goal,r_vals_window, goal_cost, barrier_cost, control_cost, h_flat
    
    
    noise_norm = []
    for i in range(8):  # numero iterazioni locali
        
        sigma_0 = args.initial_sigma
        gamma = 0.66
        sigma_k = sigma_0 * jnp.exp(-gamma * i)
        noise_norm.append(sigma_k)
        for t_start in range(0, H - L + 1, L // 2):
            t_end = t_start + L
            U_window = U[t_start:t_end]
            rng, rng_step = jax.random.split(rng)

            U_opt_local, lambda_goal,r_vals_window, goal_cost, barrier_cost, control_cost, h_flat = reverse_once_local_hybrid(U_window, rng_step,U, t_start, residual_fn,lagrangian_fn,lambda_goal,args,env,sigma_k)
            
            R_window_all.append(np.array(r_vals_window))
            J_goal_all.append(np.array(goal_cost))
            J_barrier_all.append(np.array(barrier_cost))
            J_control_all.append(np.array(control_cost))
            H_norm_all.append(np.linalg.norm(h_flat.reshape(h_flat.shape[0], -1), axis=1))
            U = U.at[t_start:t_end].set(U_opt_local)
       
        state_init_eval = reset_env_jit(jax.random.PRNGKey(args.seed + 1024))
        rewss_eval, _ = rollout_us(state_init_eval, U)

        reward_per_robot = rewss_eval.mean(axis=0)
        rewards_per_iter.append(np.array(reward_per_robot))
        reward_array_str = "[" + ", ".join(f"{r:.4f}" for r in reward_per_robot) + "]"
        print(f"[Iteration {i}] robots average rewards: {reward_array_str}")
    # print("Salvataggio iterazione", i)
    # print("Shape R_window_all[0]:", J_control_all[0].shape)
    # print("Len R_window_all:", len(J_control_all))

    np.savez("results/multicar_iterative/trend_samples_iter_7.npz",
        R_window=np.stack(R_window_all),
        J_goal=np.stack(J_goal_all),
        J_barrier=np.stack(J_barrier_all),
        J_control=np.stack(J_control_all),
        H_norm=np.stack(H_norm_all),noise_norm=np.stack(noise_norm))

        

    return U,rewards_per_iter
   

def main():
    args = tyro.cli(Args)

    total_start = time.time()

    print("STEP 1: Initial Reverse Diffusion")
    env = MultiCar2d(n=args.n_robots, formation_shift=args.formation_shift)

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

    if not args.not_render:
        path = "results/multicar_iterative"
        os.makedirs(path, exist_ok=True)


        state_init = reset_env_jit(jax.random.PRNGKey(args.seed + 1024))
        xs = jnp.array([state_init.pipeline_state])
        state = state_init
        for t in range(U_opt.shape[0]):
            state = step_env_jit(state, U_opt[t])
            xs = jnp.concatenate([xs, state.pipeline_state[None]], axis=0)
        xs = jnp.transpose(xs, (1, 0, 2))

       # Save final trajectory plot

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.set_aspect('equal', adjustable='datalim')

        env.render(ax, xs, goals=env.xg)
        
        ecd_tag =  "d4orm+ECD"
        formation_tag = "form" if args.formation_shift else ""
        plt.title(f"Optimized final trajector {ecd_tag}_{formation_tag}")
        plt.tight_layout()
        plt.savefig(os.path.join(path, f"local_diffusion_{ecd_tag}_{formation_tag}.png"))
        print(f"Figura salvata in {path}/local_diffusion.png")

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
        ax.legend(points, labels, loc='upper right')


        video_path = os.path.join(path, f"local_diffusion_{ecd_tag}_{formation_tag}.mp4")
        ani.save(video_path, fps=10, dpi=150)
        print("Video saved in:", video_path)
        


if __name__ == "__main__":
    main()



   