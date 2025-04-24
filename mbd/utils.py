import jax
from brax.io import html


# evaluate the diffused uss
def eval_us(step_env, state, us):
    def step(state, u):
        state = step_env(state, u)
        return state, state.reward

    _, rews = jax.lax.scan(step, state, us)
    return rews

def rollout_us(step_env, state, us):
    def step(state, u):
        state = step_env(state, u)
        return state, (state.reward, state.pipeline_state)

    _, (rews, pipline_states) = jax.lax.scan(step, state, us)
    return rews, pipline_states


def render_us(step_env, sys, state, us):
    rollout = []
    rew_sum = 0.0
    Hsample = us.shape[0]
    for i in range(Hsample):
        rollout.append(state.pipeline_state)
        state = step_env(state, us[i])
        rew_sum += state.reward
    # rew_mean = rew_sum / (Hsample)
    # print(f"evaluated reward mean: {rew_mean:.2e}")
    return html.render(sys, rollout)


def rollout_multi_us(step_env, state, us):
    """
    Esegue il rollout per sistemi multi-robot.

    Args:
        step_env: funzione  step dello scenario (env.step)
        state: stato iniziale
        us: controlli, shape (H, n, 2)

    Returns:
        rews: reward a ogni istante, shape (H,)
        pipeline_states: stati completi, shape (H, n, 3)
    """
    def step(state, u_t):
        # u_t ha shape (n, 2), azioni di tutti i robot all'istante t
        state = step_env(state, u_t)
        return state, (state.reward, state.pipeline_state)

    _, (rews, pipeline_states) = jax.lax.scan(step, state, us)
    return rews, pipeline_states
