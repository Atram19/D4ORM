from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import tyro

from mbd.envs.class_manipulator import RRPRSingleEnv, Args, rollout_single_us, forward_kinematics_rrpr_jax

def run_diffusion_once(args: Args, env, rollout_us, reset_env_jit):
    rng = jax.random.PRNGKey(seed=args.seed)

    Nx = env.observation_size
    Nu = env.action_size

    rng, rng_reset = jax.random.split(rng)
    state_init = reset_env_jit(rng_reset)

    # Diffusion noise schedule
    betas = jnp.linspace(args.beta0, args.betaT, args.Ndiffuse)
    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)
    sigmas = jnp.sqrt(1 - alphas_bar)

    YN = jnp.zeros([args.Hsample, Nu])

    @jax.jit
    def reverse_once(carry):
        i, rng, Ybar_i = carry
        Yi = Ybar_i * jnp.sqrt(alphas_bar[i])

        rng, rng_eps = jax.random.split(rng)
        eps_u = jax.random.normal(rng_eps, (args.Nsample, args.Hsample, Nu))
        Y0s = eps_u * sigmas[i] + Ybar_i

        rewss, pipeline_state, r_terms = jax.vmap(rollout_us)(Y0s)
        rews = rewss.mean(axis=-1)

        rew_std = rews.std()
        rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std)

        final_ee = pipeline_state[:, -1, :4]
        T_final, *_ = jax.vmap(lambda q: forward_kinematics_rrpr_jax(
            q, env.L1_num, env.L2_num, env.L3_num, env.L4_num, env.D2_num))(final_ee)
        ee_final_pos = T_final[:, :3, 3]

        T_goal, *_ = forward_kinematics_rrpr_jax(env.qf, env.L1_num, env.L2_num,
                                                 env.L3_num, env.L4_num, env.D2_num)
        goal_pos = T_goal[:3, 3]

        dist = jnp.linalg.norm(ee_final_pos - goal_pos[None, :], axis=-1)
        penalty = 10.0 * dist

        logp0 = (rews - rews.mean()) / rew_std / args.temp_sample
        # logp0 -= penalty  # opzionale: penalizzare distanza da goal

        weights = jax.nn.softmax(logp0)
        Ybar = jnp.einsum("s,shj->hj", weights, Y0s)

        score = 1 / (1.0 - alphas_bar[i]) * (-Yi + jnp.sqrt(alphas_bar[i]) * Ybar)
        Yim1 = 1 / jnp.sqrt(alphas[i]) * (Yi + (1.0 - alphas_bar[i]) * score)
        Ybar_im1 = Yim1 / jnp.sqrt(alphas_bar[i - 1])

        return (i - 1, rng, Ybar_im1), Yi, Y0s

    def reverse(YN, rng):
        Yi = YN
        for i in reversed(range(1, args.Ndiffuse)):
            carry = (i, rng, Yi)
            (i, rng, Yi), Yi_current, Y0s = reverse_once(carry)
        return Yi

    rng_exp, rng = jax.random.split(rng)
    U_0 = reverse(YN, rng_exp)
    return U_0

def main():
    args = tyro.cli(Args)

    print("STEP 1: Initial Reverse Diffusion")
    env = RRPRSingleEnv()

    step_env_jit = jax.jit(env.step)
    reset_env_jit = jax.jit(env.reset)
    state_init = reset_env_jit(jax.random.PRNGKey(args.seed))

    rollout_us_fn = partial(rollout_single_us, step_env_jit, state_init)

    t1 = time.time()
    U_init = run_diffusion_once(args, env, rollout_us_fn, reset_env_jit)
    t2 = time.time()
    print(f"Initial reverse diffusion time: {t2 - t1:.3f} s")

    rewards, x_traj, r_terms = rollout_us_fn(U_init)
    path = "results/rrpr_manipolator"
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

    env.render(x_init, tau_seq=U_init, rewards=rewards, r_terms=r_terms)

if __name__ == "__main__":
    main()