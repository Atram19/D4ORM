from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import tyro
import jax.debug
from mbd.envs.class_manipulator import RRPRSingleEnv, Args, rollout_single_us, forward_kinematics_rrpr_jax
def compute_metrics(env, x_traj, tau_seq, rewards, goal_xyz, tag=""):
    # Stati e torques
    x_np = np.array(x_traj)
    tau_np = np.array(tau_seq)

    # === Metriche in joint space (diagnostica) ===
    q = x_np[:, :4]
    qf = np.array(env.qf[:4])
    joint_err = q - qf
    rmse_joints = np.sqrt(np.mean(joint_err**2, axis=0))
    mae_joints  = np.mean(np.abs(joint_err), axis=0)
    maxerr_joints = np.max(np.abs(joint_err), axis=0)

    # === Traiettoria end-effector ===
    ee_traj = []
    for qt in q:
        T_curr, *_ = forward_kinematics_rrpr_jax(
            qt, env.L1_num, env.L2_num, env.L3_num, env.L4_num, env.D2_num
        )
        ee_traj.append(np.array(T_curr[:3, 3]))
    ee_traj = np.array(ee_traj)   # (T, 3)

    # Errori EE
    ee_err = ee_traj - goal_xyz   # (T, 3)
    rmse_ee = np.sqrt(np.mean(ee_err**2, axis=0))
    mae_ee  = np.mean(np.abs(ee_err), axis=0)
    final_ee_error = ee_err[-1]
    final_ee_norm  = np.linalg.norm(final_ee_error)

    # === Sforzo di controllo (tau reali) ===
    tau_scaled = tau_np * np.array(env.ACTION_SCALE)
    energy_control = np.sum(np.linalg.norm(tau_scaled, axis=1)**2) * env.dt
    peak_tau = np.max(np.abs(tau_scaled))

    # === Tempo di assestamento (ultimo ingresso sotto 5% errore iniziale EE) ===
    ee_norm_errors = np.linalg.norm(ee_err, axis=1)
    thresh = 0.05 * ee_norm_errors[0]
    above = np.where(ee_norm_errors >= thresh)[0]
    if above.size > 0:
        last_out = np.max(above)
        Ts = (last_out + 1) * env.dt
    else:
        Ts = 0.0

    # === Reward totale ===
    R_total = float(np.sum(np.array(rewards)))

    metrics = {
        # --- End-effector (principali) ---
        "RMSE EE [x,y,z]": rmse_ee,
        "MAE EE [x,y,z]": mae_ee,
        "Final EE error [x,y,z]": final_ee_error,
        "Final EE norm [m]": final_ee_norm,
        # --- Control effort ---
        "Energy control": energy_control,
        "Peak torque": peak_tau,
        # --- Tempo ---
        "Settling time [s]": Ts,
        # --- Reward ---
        "Total reward": R_total,
        # --- Joint space (diagnostica) ---
        "RMSE joints": rmse_joints,
        "MAE joints": mae_joints,
        "Max error joints": maxerr_joints,
    }

    # Salva su file
    os.makedirs("results/manipulator", exist_ok=True)
    with open(f"results/manipulator/metrics_{tag}.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    return metrics


def cosine_beta_schedule(T, s=0.008):
    t = jnp.arange(T + 1, dtype=jnp.float32)
    f_t = jnp.cos(((t / T + s) / (1 + s)) * jnp.pi / 2) ** 2
    alphas_bar = f_t / f_t[0]
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    betas = 1 - alphas
    return jnp.clip(betas, 1e-5, 0.999)

def run_diffusion_once(args: Args, env, rollout_us, reset_env_jit):
    rng = jax.random.PRNGKey(seed=args.seed)

    Nx = env.observation_size
    Nu = env.action_size

    rng, rng_reset = jax.random.split(rng)
    state_init = reset_env_jit(rng_reset)

    # Diffusion noise schedule
    betas = jnp.linspace(args.beta0, args.betaT, args.Ndiffuse)
    #betas = cosine_beta_schedule(args.Ndiffuse)

    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)
    sigmas = jnp.sqrt(1 - alphas_bar)

    Y0s_list = []
    states_xyz_all = [] 
    YN = jnp.zeros([args.Hsample, Nu])

    #@jax.jit
    def reverse_once(carry):
        i, rng, Ybar_i = carry
        Yi = Ybar_i * jnp.sqrt(alphas_bar[i])

        rng, rng_eps = jax.random.split(rng)
        eps_u = jax.random.normal(rng_eps, (args.Nsample, args.Hsample, Nu))

        
        Y0s = eps_u * sigmas[i] + Ybar_i
        Y0s = jnp.clip(Y0s,-1,1)
        #Y0s_list.append(np.array(Y0s))

        # == Calcola e salva le traiettorie xyz per i primi Nplot sample a questo step ==
        Nplot = 10 
        rewss, pipeline_state, r_terms = jax.vmap(rollout_us)(Y0s)
        pipeline_plot = pipeline_state[:Nplot]
        states_xyz_all.append(np.array(pipeline_plot))
        rews = rewss.mean(axis=-1)

        rew_std = rews.std()
        rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std)

        # effective_temp = jnp.maximum(0.01, args.temp_sample * (i / args.Ndiffuse + 1e-5))
        # logp0 = (rews - rews.mean()) / rew_std / effective_temp

        logp0 = (rews - rews.mean()) / rew_std / args.temp_sample

        weights = jax.nn.softmax(logp0)
        Ybar = jnp.einsum("s,shj->hj", weights, Y0s)

        score = 1 / (1.0 - alphas_bar[i]) * (-Yi + jnp.sqrt(alphas_bar[i]) * Ybar)
        Yim1 = 1 / jnp.sqrt(alphas[i]) * (Yi + (1.0 - alphas_bar[i]) * score)
        Ybar_im1 = Yim1 / jnp.sqrt(alphas_bar[i - 1])
        
        jax.debug.print("Step {}: mean reward = {:.4f}, std = {:.4f}", i, rews.mean(), rew_std)

        return (i - 1, rng, Ybar_im1), Yi, Y0s
  
    def reverse(YN, rng):
        Yi = YN
        for i in reversed(range(1, args.Ndiffuse)):
            carry = (i, rng, Yi)
            (i, rng, Yi), Yi_current, Y0s = reverse_once(carry)
        np.savez("results/rrpr_states_over_steps.npz", states=np.array(states_xyz_all))
        return Yi
 


    rng_exp, rng = jax.random.split(rng)
    U_0 = reverse(YN, rng_exp)
   
    return U_0

def run_diffusion_local(args: Args, U_init: jnp.ndarray, env, rollout_us, reset_env_jit):
    rng = jax.random.PRNGKey(seed=args.seed + 123)
    rewards_per_iter = []

    H = args.Hsample
    Nu = env.action_size

    U = U_init.copy()
    U_ks = [np.array(U_init)]  # salva le traiettorie U dopo ogni iterazione (inclusa quella globale iniziale)

    L = 10  # window length
    K = 1  # local iterations

    betas = jnp.linspace(args.beta0, args.betaT, L)
    alphas = 1.0 - betas
    alphas_bar_local = jnp.cumprod(alphas)
    sigmas_local = jnp.sqrt(1 - alphas_bar_local)
    frequenze_k = []
    tempi_k = []
    states_xyz_local = []
    tempi_finestra = []
    for k in range(K):
        t_k_start = time.time()
        samples_k = []  # lista delle finestre per iterazione k

        for t_start in range(0, H - L + 1, L // 2):
            t_end = t_start + L
            U_window = U[t_start:t_end]
            start_win = time.time()
            rng, rng_step = jax.random.split(rng)
            #@jax.jit
            def reverse_once_local(U_w, rng_w):
                for j in reversed(range(1, L)):
                    eps_u = jax.random.normal(rng_w, (args.Nsample, L, Nu))
                    sigma_local = sigmas_local[j]
                    Y0s = eps_u * sigma_local + U_w  

                    Y0s = jnp.clip(Y0s, -1, 1)  # Clip to action bounds

                    # Insert Y0s into full trajectory
                    U_fulls = jnp.repeat(U[None, ...], args.Nsample, axis=0)  # (Nsample, H, Nu)
                    U_fulls = U_fulls.at[:, t_start:t_end, :].set(Y0s)
                    Nplot_local = 20
                    state_init = reset_env_jit(rng_step)
                    rewss, pipeline_local,_ = jax.vmap(rollout_us)(U_fulls)
                    rews = rewss.mean(axis=-1)  
                    pipeline_plot_local = pipeline_local[:Nplot_local]
                    rews = rewss.mean(axis=-1)

                    logp0 = (rews - rews.mean()) / (rews.std() + 1e-6) / args.temp_sample
                    weights = jax.nn.softmax(logp0)
                    U_opt = jnp.einsum("s,slj->lj", weights, Y0s)

                    U_new = jnp.sqrt(alphas_bar_local[j - 1]) * U_opt
                    #U_w = U_new

                return U_new,pipeline_plot_local 

            #U_opt_local = reverse_once_local(U_window, rng_step)
            start = time.time()
            U_opt_local,pipeline_plot_local  = reverse_once_local(U_window, rng_step)    
            end = time.time()
            delta = end - start

            # Speed ratio
            speed_ratio = (L * env.dt) / delta
            print(f"Finestra [{t_start}:{t_end}] - tempo = {delta:.3f} s - speed ratio = {speed_ratio:.3f}")
            # misura tempo fine
            end_win = time.time()
            delta_win = end_win - start_win     
            tempi_finestra.append(delta_win)   # accumula tempi singole finestre   
            samples_k.append(np.array(pipeline_plot_local))  # salvi campioni di questa finestra
            # jax.block_until_ready(U_opt_local)
            # end = time.time()
            # print(f"Tempo reale: {end - start:.2f} s")
            U = U.at[t_start:t_end, :].set(U_opt_local)
      #  U_ks.append(np.array(U))
     #   states_xyz_local.append(samples_k)  # salvi tutta l'iterazione k


        t_k_end = time.time()
        tempo_k = t_k_end - t_k_start
        frequenza_k = 1.0 / tempo_k if tempo_k > 0 else 0.0
        tempi_k.append(tempo_k)
        
        frequenze_k.append(frequenza_k)
        state_init_eval = reset_env_jit(jax.random.PRNGKey(args.seed ))
        rewss_eval, _ ,_= rollout_us(U)
        reward_mean = rewss_eval.mean()
        rewards_per_iter.append(float(reward_mean))
        tempo_medio_finestra = np.mean(tempi_finestra)
        print(f"[Iteration {k}] tempo medio finestra = {tempo_medio_finestra:.4f} s")
        print(f"[Iteration {k}] reward = {reward_mean:.4f} | tempo = {tempo_k:.3f}s | freq = {frequenza_k:.2f} Hz")
    # np.savez("results/rrpr_states_local.npz", states_local=np.array(states_xyz_local))
    # np.savez("results/rrpr_U_local.npz", U_ks=np.array(U_ks))  # shape (K+1, H, Nu)

    return U, rewards_per_iter


def main():
    args = tyro.cli(Args)
    print(jax.devices())
    print("STEP 1: Initial Reverse Diffusion")
    env = RRPRSingleEnv(dt = 0.005)

    step_env_jit = jax.jit(env.step)
    reset_env_jit = jax.jit(env.reset)
    state_init = reset_env_jit(jax.random.PRNGKey(args.seed))

    rollout_us_fn = jax.jit(partial(rollout_single_us, step_env_jit, state_init))

    t1 = time.time()
    U_init = run_diffusion_once(args, env, rollout_us_fn, reset_env_jit)
    t2 = time.time()
    print(f"Initial reverse diffusion time: {t2 - t1:.3f} s")
    t3 = time.time()
    U_optimized, rewards_per_iter = run_diffusion_local(args=args,U_init=U_init,env=env,rollout_us=rollout_us_fn,reset_env_jit=reset_env_jit)
    t4 = time.time()
    print(f"Local optimization time: {t4 - t3:.3f} s")
    rewards_opt, x_traj, r_terms_opt = rollout_us_fn(U_optimized)
    rewards, x_traj, r_terms = rollout_us_fn(U_init)
    path = "results/manipulator"
    os.makedirs(path, exist_ok=True)

    state = reset_env_jit(jax.random.PRNGKey(args.seed))
    states = []
    for t in range(U_init.shape[0]):
        u_t = U_init[t]
        state = step_env_jit(state, u_t)
        states.append(state.pipeline_state)

    x_init = jnp.stack(states, axis=0)

    T_goal, *_ = forward_kinematics_rrpr_jax(env.qf, env.L1_num, env.L2_num,
                                             env.L3_num, env.L4_num, env.D2_num)
    goal_xyz = np.array(T_goal[:3, 3])
    np.savez("results/goal_xyz.npz", goal=goal_xyz)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_aspect('equal', adjustable='datalim')

    state = reset_env_jit(jax.random.PRNGKey(args.seed))
    states = []
    for t in range(U_init.shape[0]):
        u_t = U_optimized[t]
        state = step_env_jit(state, u_t)
        states.append(state.pipeline_state)

    x_opt = jnp.stack(states, axis=0)
    fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5))
    ax1.set_aspect('equal', adjustable='datalim')

    env.render(x_init, tau_seq=U_init, rewards=rewards, r_terms=r_terms, tag = "global")
    env.render(x_opt, tau_seq=U_optimized, rewards=rewards_opt, r_terms=r_terms_opt, tag = "local")
    np.savez("x_opt_rrpr.npz", x_opt=x_opt, goal=goal_xyz)
        # === Calcolo metriche ===
    metrics_global = compute_metrics(env, x_init, U_init, rewards, goal_xyz, tag="global")
    metrics_local  = compute_metrics(env, x_opt, U_optimized, rewards_opt, goal_xyz, tag="local")

    print("\n=== Metriche Globali ===")
    for k,v in metrics_global.items():
        print(k, ":", v)
    print("\n=== Metriche Locali ===")
    for k,v in metrics_local.items():
        print(k, ":", v)




if __name__ == "__main__":
    main()