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



def run_diffusion_once(args: Args, env, rollout_us, reset_env_jit):
    """
    First phase of D4ORM: initial global reverse diffusion (Algorithm 1).
    Now also saves the (x, y) rollout trajectories and Ybar at each reverse step.
    """

    rng = jax.random.PRNGKey(seed=args.seed)

    Nx = env.observation_size
    Nu = env.action_size
    n = env.num_robots
    T = 99  # reverse steps  HO AUMENTATO A 59 INVECE DI 19 

    rng, rng_reset = jax.random.split(rng)
    state_init = reset_env_jit(rng_reset)

    # Diffusion noise schedule
    betas = jnp.linspace(args.beta0, args.betaT, 100)
    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)
    sigmas = jnp.sqrt(1 - alphas_bar)

    
    # Start from zero control (like UN ~ N(0,I))
    YN = jnp.zeros([args.Hsample, n, Nu])
    


    # Preallocate buffer for (x, y) trajectories and Ybar
    def init_buffers():
        return (
            jnp.zeros((T, args.Nsample, args.Hsample, n, 2)),  # sample_trajectories_xy
            jnp.zeros((T, args.Hsample, n, Nu)),                # Ybar_list
            jnp.zeros((T, args.Nsample, n, 6))                 # reward_terms
        )

    # Jitted single reverse diffusion step
    @jax.jit
    def reverse_once(carry, _):
        i, rng, Ybar_i, traj_buf, ybar_buf,reward_buf = carry
        Yi = Ybar_i * jnp.sqrt(alphas_bar[i])

        rng, rng_eps = jax.random.split(rng)
        eps_u = jax.random.normal(rng_eps, (args.Nsample, args.Hsample, n, Nu))
        

        Y0s = eps_u * sigmas[i] + Ybar_i
        Y0s = jnp.clip(Y0s, -1.0, 1.0)

        # rollout of all sample controls
        rewss, pipeline_states = jax.vmap(rollout_us, in_axes=(None, 0))(state_init, Y0s)
        rews = rewss.mean(axis=(1, 2))

        trajs_xy = pipeline_states[..., :2]  # (Nsample, H, n, 2)
        traj_buf = traj_buf.at[i - 1].set(trajs_xy)
        # Reward terms finali (usando l’ultima q e u)
        q_final = pipeline_states[:, -1]   # (Nsample, n, Nx)
        u_final = Y0s[:, -1]               # (Nsample, n, Nu)
        reward_terms = jax.vmap(env.get_reward_terms)(q_final, u_final)  # (Nsample, n, 6)
        reward_buf = reward_buf.at[i - 1].set(reward_terms)
        # score
        rew_std = rews.std()
        rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std)
        #temp_schedule = jnp.linspace(1.0, 0.1, args.Ndiffuse)  # o 100 se hardcoded

        logp0 = (rews - rews.mean()) / rew_std / args.temp_sample
        weights = jax.nn.softmax(logp0)

        Ybar = jnp.einsum("s,shij->hij", weights, Y0s)
        ybar_buf = ybar_buf.at[i - 1].set(Ybar)

        score = 1 / (1.0 - alphas_bar[i]) * (-Yi + jnp.sqrt(alphas_bar[i]) * Ybar)
        Yim1 = 1 / jnp.sqrt(alphas[i]) * (Yi + (1.0 - alphas_bar[i]) * score)
        Ybar_im1 = Yim1 / jnp.sqrt(alphas_bar[i - 1])

        return (i - 1, rng, Ybar_im1, traj_buf, ybar_buf,reward_buf), None

    def reverse(YN, rng):
        traj_buf, ybar_buf,reward_buf = init_buffers()
        carry = (T, rng, YN, traj_buf, ybar_buf,reward_buf)
        (i_final, rng_final, U_0, traj_buf, ybar_buf,reward_buf), _ = jax.lax.scan(reverse_once, carry, None, length=T)
        return U_0, traj_buf, ybar_buf,reward_buf

    rng_exp, rng = jax.random.split(rng)
    U_0, sample_trajectories_xy, Ybar_list,reward_buf = reverse(YN, rng_exp)

    # Final evaluation
    state_init_eval = reset_env_jit(jax.random.PRNGKey(args.seed + 1024))
    rewss_eval, _ = rollout_us(state_init_eval, U_0)
    reward_per_robot = rewss_eval.mean(axis=0)

    reward_array_str = "[" + ", ".join(f"{r:.4f}" for r in reward_per_robot) + "]"
    print(f"global robots average rewards: {reward_array_str}")
    # Compute reward terms along the entire trajectory Ybar = U_0
    _, q_traj_opt = rollout_us(state_init_eval, U_0)  # shape (H, n, Nx)
    u_traj_opt = U_0                                 # shape (H, n, Nu)
    reward_traj_opt = env.get_reward_trajectory_terms(q_traj_opt, u_traj_opt)  # (H, n, 6)

    # Salvataggio
    np.savez("results/multicar_iterative/global_diffusion_data.npz",
         sample_trajectories_xy=np.array(sample_trajectories_xy),
         Ybar_list=np.array(Ybar_list),
         reward_terms=np.array(reward_buf),
         reward_traj_opt=np.array(reward_traj_opt),
         U_0=np.array(U_0))


    return U_0

def run_diffusion_local(args: Args, U_init: jnp.ndarray, env, rollout_us, reset_env_jit):
    """
    Second phase of D4ORM: local iterative reverse diffusion optimization.
    Based on Algorithm 2 (Iterative Denoising) from the D4ORM paper.
    Now also saves rollout trajectories (x, y) for visualization.
    """

    rng = jax.random.PRNGKey(seed=args.seed + 123)
    rewards_per_iter = []
    H = args.Hsample
    Nu = env.action_size
    n = env.num_robots

    U = U_init.copy()

    # Local diffusion parameters
    L = 10  # window length
    K = 5   # number of local iterations
    betas_local = jnp.linspace(0.01, 0.2, 10)
    alphas_local = 1.0 - betas_local
    alphas_bar_local = jnp.cumprod(alphas_local)
    sigmas_local = jnp.sqrt(1 - alphas_bar_local)
    sigma_local = sigmas_local[-1]

    lambda_goal = jnp.zeros((env.n * 2,))

    # Funzioni ausiliarie
    state_init_for_goal = reset_env_jit(jax.random.PRNGKey(args.seed +777))
    residual_fn = make_residual_fn(state_init_for_goal, env, args.Nsample)
    lagrangian_fn = make_lagrangian_fn(state_init_for_goal, env, args.Nsample)

  
    
    def reverse_once_local_hybrid(U_window, rng_w, U_full_template, t_start, residual_fn, lagrangian_fn, lambda_goal, args, env,sigma_local):
        Nsample = args.Nsample
        L, n, Nu = U_window.shape
        sigma_local=sigma_local

        # Step 1: Sampling
        eps_u = jax.random.normal(rng_w, (Nsample, L, n, Nu))
        Y0s = eps_u * sigma_local + U_window
        noise = eps_u * sigma_local
        Y0s = jnp.clip(Y0s, -1.0, 1.0)

        # Step 2: Rollouts completi
        
        U_fulls = jnp.repeat(U_full_template[None, ...], Nsample, axis=0)
        U_fulls = U_fulls.at[:, t_start:t_start+L, :, :].set(Y0s)

        Y0s_window = Y0s[:, t_start:t_start+L, :, :]
        state_init = env.reset(jax.random.PRNGKey(args.seed + 1024))
        rewss, pipeline_states = jax.vmap(rollout_us, in_axes=(None, 0))(state_init, U_fulls)
        pipeline_states_window = pipeline_states[:, t_start:t_start+L]
        
        r_vals_window = rewss[:, t_start:t_start+L, :].mean(axis=(1,2))
        
        
        rews = rewss.mean(axis=(1,2))
        # Step 3: Lagrangiana
        L_cost, L_constraint, L_tot, control_cost, barrier_cost, goal_cost, h_flat,obstacles = lagrangian_fn(
            Y0s_window, Y0s, pipeline_states, pipeline_states_window, lambda_goal, args.mu)

       
        score_vals = 10*rews-L_tot
        print(f"[D4orm {i}] L_vals ∈ [{L_tot.min():.4f}, {L_tot.max():.4f}], rews_k ∈ [{rews.min():.4f}, {rews.max():.4f}]")

        logp0 = (score_vals -score_vals.mean()) / (score_vals.std() + 1e-6) / args.temp_sample
        weights = jax.nn.softmax(logp0)
        U_soft = jnp.einsum("s,slij->lij", weights, Y0s)
        


        return U_soft , r_vals_window, goal_cost, barrier_cost, control_cost, h_flat,obstacles,L_cost,L_tot

    def ecd_refinement(U_start, lambda_start, rews):
        H, n, Nu = U_start.shape
        Nsample = args.Nsample
        U_curr = U_start
        lambda_curr = lambda_start
        U_soft_single = U_start  # shape (L, n, Nu)
        
        state_soft = env.reset(jax.random.PRNGKey(args.seed + 1024))
        rew_soft, pipeline_soft = rollout_us(state_soft, U_soft_single)
        U_soft_batch = jnp.repeat(U_soft_single[None, ...], args.Nsample, axis=0)  # shape (Nsample, L, n, Nu)
        state_soft = reset_env_jit(jax.random.PRNGKey(args.seed + 1024))
        #rew_soft, pipeline_soft = jax.vmap(rollout_us, in_axes=(None, 0))(state_soft, U_soft_batch)

        
        # Usa una nuova lagrangiana con batch size = 1
        lagrangian_fn_1 = make_lagrangian_fn(state_soft, env, Nsample=1)

        L_soft, _, L_tot_soft, *_ = lagrangian_fn_1(
            U_soft_single[None, ...], U_soft_single[None, ...], 
            pipeline_soft[None, ...], pipeline_soft[None, ...], 
            lambda_curr, args.mu
        )

        baseline = -(rew_soft.mean() - L_tot_soft[0] )
        print(f"   baseline = {baseline:.4f}")

        for i in range(30):
            sigma_k = args.initial_sigma * jnp.exp(-args.noise_decay * i)
            mu_k = args.mu

            rng_key = jax.random.PRNGKey(args.seed + 1024 + i)
            eps_u = jax.random.normal(rng_key, (Nsample, H, n, Nu))
            noise = eps_u * sigma_k
            Y0s_k =jnp.clip( U_curr + noise,-1,1) # Sample attorno a U_curr
            # Debug 1: Controllo sui valori delle azioni
            print(f"\n[ECD {i}] --- sigma_k = {sigma_k:.6f}")
            print(f"[ECD {i}] Y0s_k ∈ [{Y0s_k.min():.3f}, {Y0s_k.max():.3f}]")
            # Subito dopo il clipping di Y0s_k
            # max_val = Y0s_k.max()
            # min_val = Y0s_k.min()
            # n_clipped_max = jnp.sum(Y0s_k == 1.0)
            # n_clipped_min = jnp.sum(Y0s_k == -1.0)
            # print(f"[ECD {i}] Y0s_k ∈ [{min_val:.3f}, {max_val:.3f}], clipped +1: {n_clipped_max}, clipped -1: {n_clipped_min}")

            # Rollouts
            state_k = reset_env_jit(rng_key)
            rewss_k, pipeline_states_k = jax.vmap(rollout_us, in_axes=(None, 0))(state_k, Y0s_k)

            rews_k = rewss_k.mean(axis=(1, 2))  # shape (Nsample,)
           
            #rews_k = (rews_k - rews_k.mean()) / (rews_k.std() + 1e-6)

            # Lagrangiana su tutta la traiettoria
            L_cost, L_constraint, L_vals_k, control_cost, barrier_cost, goal_cost, h_flat_k, _ = \
                lagrangian_fn(Y0s_k, Y0s_k, pipeline_states_k, pipeline_states_k, lambda_curr, mu_k)
            #L_vals_k = (L_vals_k - L_vals_k.mean()) / (L_vals_k.std() + 1e-6)
            # Gradient REINFORCE
            print(f"[ECD {i}] ||h|| = {jnp.linalg.norm(h_flat_k):.3f}")

            #stim =  L_vals_k/1000
            alpha_r = 1.0   # peso reward
            alpha_l = 1e-3  # peso lagrangiana (scalata)

            stim = -1*(alpha_r * rews_k - alpha_l * L_vals_k)
            print(f"[ECD {i}] stim ∈ [{stim.min():.4f}, {stim.max():.4f}]")
            print(f"[ECD {i}] L_vals ∈ [{L_vals_k.min():.4f}, {L_vals_k.max():.4f}], rews_k ∈ [{rews_k.min():.4f}, {rews_k.max():.4f}]")

            grad = jnp.einsum("s,slij->lij",stim-stim.mean() , eps_u)
            grad = grad / (args.Nsample * sigma_k ** 2 )

            # Debug 3: Norme del gradiente
            grad_norm = jnp.linalg.norm(grad)
            print(f"[ECD {i}] ||grad|| = {grad_norm:.3e}")
            # U_next = U_curr - args.alpha * grad 
            # Rumore Langevin per l'aggiornamento finale
            rng_key, rng_noise = jax.random.split(rng_key)
            xi = jax.random.normal(rng_noise, U_curr.shape)
            sigma_langevin = sigma_k * jnp.sqrt(2 * args.alpha)
            noise_term = sigma_langevin * xi

            U_next = U_curr - args.alpha * grad +noise_term
            U_next = jnp.clip(U_next, -1.0, 1.0)

            print(f"[ECD {i}] ||U_curr|| = {jnp.linalg.norm(U_curr):.3f}, ||U_next|| = {jnp.linalg.norm(U_next):.3f}")
            # Aggiornamento lambda
            h_goal = residual_fn(pipeline_states_k)  
            h_mean = jnp.mean(h_goal, axis=0).reshape(-1)
            lambda_next = lambda_curr + args.alpha * mu_k * h_mean
            # Update
            U_curr = U_next
            lambda_curr = lambda_next
            print(f"[ECD {i}] ||λ|| = {jnp.linalg.norm(lambda_curr):.3e}")
        return U_curr, lambda_curr

        
    
    for i in range(20):  # iterazioni locali
        for t_start in range(0, H - L + 1, L // 2):
            t_end = t_start + L
            U_window = U[t_start:t_end]
            rng, rng_step = jax.random.split(rng)

            U_opt_local, r_vals_window, goal_cost, barrier_cost, control_cost, h_flat,obstacles, L_cost,L_tot  = reverse_once_local_hybrid(
                U_window, rng_step, U, t_start, residual_fn, lagrangian_fn, lambda_goal, args, env,sigma_local)
            U = U.at[t_start:t_end].set(U_opt_local)
            
           

        state_eval = reset_env_jit(jax.random.PRNGKey(args.seed + 1024 + i))
        _, pipeline_states = rollout_us(state_eval, U)
        pipeline_states = pipeline_states[None, ...] 
        residual_fn_1 = make_residual_fn(state_init_for_goal, env, Nsample=1)   
        h_global = residual_fn_1(pipeline_states)
        h_mean = jnp.mean(h_global, axis=0).reshape(-1)
        lambda_goal = lambda_goal + args.alpha * args.mu * h_mean

        # Valutazione reward finale
        state_init_eval = reset_env_jit(jax.random.PRNGKey(args.seed + 1024))
        rewss_eval, _ = rollout_us(state_init_eval, U)
        reward_per_robot = rewss_eval.mean(axis=0)
        rewards_per_iter.append(np.array(reward_per_robot))
        reward_array_str = "[" + ", ".join(f"{r:.4f}" for r in reward_per_robot) + "]"
        print(f"[Iteration {i}] robots average rewards: {reward_array_str}")
    lambda_new = jnp.zeros((env.n * 2,))
    U_refined, lambda_goal = ecd_refinement(U, lambda_new, rews=None)
    state_init_eval = reset_env_jit(jax.random.PRNGKey(args.seed + 1024))
    rewss_eval, _ = rollout_us(state_init_eval, U_refined)
    reward_per_robot = rewss_eval.mean(axis=0)
    rewards_per_iter.append(np.array(reward_per_robot))
    reward_array_str = "[" + ", ".join(f"{r:.4f}" for r in reward_per_robot) + "]"
    print(f"[Iteration ECD] robots average rewards: {reward_array_str}")
    return U_refined, rewards_per_iter


def main():
    args = tyro.cli(Args)

    total_start = time.time()

    print("STEP 1: Initial Reverse Diffusion")
    env = MultiCar2d(n=args.n_robots, formation_shift=args.formation_shift,obstacles_enabled=args.obstacles_enabled,ECD= args.ECD)

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



   