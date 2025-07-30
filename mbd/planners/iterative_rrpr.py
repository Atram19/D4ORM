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
        Y0s_list.append(np.array(Y0s))

        # == Calcola e salva le traiettorie xyz per i primi Nplot sample a questo step ==
        Nplot = 10 
        rewss, pipeline_state, r_terms = jax.vmap(rollout_us)(Y0s)
        pipeline_plot = pipeline_state[:Nplot]
        states_xyz_all.append(np.array(pipeline_plot))
        rews = rewss.mean(axis=-1)

        rew_std = rews.std()
        rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std)

        # final_ee = pipeline_state[:, -1, :4]
        # T_final, *_ = jax.vmap(lambda q: forward_kinematics_rrpr_jax(
        #     q, env.L1_num, env.L2_num, env.L3_num, env.L4_num, env.D2_num))(final_ee)
        # ee_final_pos = T_final[:, :3, 3]

        # T_goal, *_ = forward_kinematics_rrpr_jax(env.qf, env.L1_num, env.L2_num,
        #                                          env.L3_num, env.L4_num, env.D2_num)
        # goal_pos = T_goal[:3, 3]

        # dist = jnp.linalg.norm(ee_final_pos - goal_pos[None, :], axis=-1)
        # penalty = 10.0 * dist

        logp0 = (rews - rews.mean()) / rew_std / args.temp_sample
        # logp0 -= penalty  # opzionale: penalizzare distanza da goal

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

    L = 10  # window length
    K = 10  # local iterations

    betas = jnp.linspace(args.beta0, args.betaT, L)
    alphas = 1.0 - betas
    alphas_bar_local = jnp.cumprod(alphas)
    sigmas_local = jnp.sqrt(1 - alphas_bar_local)

    for k in range(K):
        for t_start in range(0, H - L + 1, L // 2):
            t_end = t_start + L
            U_window = U[t_start:t_end]

            rng, rng_step = jax.random.split(rng)

            def reverse_once_local(U_w, rng_w):
                for j in reversed(range(1, L)):
                    eps_u = jax.random.normal(rng_w, (args.Nsample, L, Nu))
                    sigma_local = sigmas_local[j]
                    Y0s = eps_u * sigma_local + U_w  # (Nsample, L, Nu)

                    Y0s = jnp.clip(Y0s, -1, 1)  # Clip to action bounds

                    # Insert Y0s into full trajectory
                    U_fulls = jnp.repeat(U[None, ...], args.Nsample, axis=0)  # (Nsample, H, Nu)
                    U_fulls = U_fulls.at[:, t_start:t_end, :].set(Y0s)

                    state_init = reset_env_jit(rng_step)
                    rewss, _,_ = jax.vmap(rollout_us)(U_fulls)
                    rews = rewss.mean(axis=-1)  # (Nsample,)

                    logp0 = (rews - rews.mean()) / (rews.std() + 1e-6) / args.temp_sample
                    weights = jax.nn.softmax(logp0)
                    U_opt = jnp.einsum("s,slj->lj", weights, Y0s)

                    U_new = jnp.sqrt(alphas_bar_local[j - 1]) * U_opt

                return U_new

            U_opt_local = reverse_once_local(U_window, rng_step)
            U = U.at[t_start:t_end, :].set(U_opt_local)

        state_init_eval = reset_env_jit(jax.random.PRNGKey(args.seed + 1024))
        rewss_eval, _ ,_= rollout_us(U)
        reward_mean = rewss_eval.mean()
        rewards_per_iter.append(float(reward_mean))
        print(f"[Iteration {k}] reward = {reward_mean:.4f}")

    return U, rewards_per_iter


def main():
    args = tyro.cli(Args)

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
    U_optimized, rewards_per_iter = run_diffusion_local(args=args,U_init=U_init,env=env,rollout_us=rollout_us_fn,reset_env_jit=reset_env_jit)
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


if __name__ == "__main__":
    main()