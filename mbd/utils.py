import jax
from brax.io import html
from jax import numpy as jnp

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
    rollout for multi-robot systems.

    Args:
        step_env: environment step function (env.step)
        state: initial state
        us: controls, shape (H, n, 2)

    Returns:
        rews: rewards at each timestep, shape (H,)
        pipeline_states: full state trajectories, shape (H, n, 3)
    """
    def step(state, u_t):
        
        state = step_env(state,u_t)
        return state, (state.reward, state.pipeline_state)

    _, (rews, pipeline_states) = jax.lax.scan(step, state, us)
    return rews, pipeline_states
    



 # === For ECD optimization ===


# Quadratic cost function on U
@jax.jit
def cost_fn(U_seq):
    """
    Somma dei quadrati dei controlli su batch di sequenze U_seq
    """
    return jnp.sum(U_seq ** 2, axis=(1, 2, 3))

def make_reverse_penalty_cost():
    """
    Penalizza la retromarcia (v < 0) con penalizzazione quadratica.
    """
    def sample_penalty(U_seq):  
        v = U_seq[:, :, 1]  # velocità lineare
        retro = jnp.minimum(v, 0)
        penalty = 50* jnp.mean(retro ** 2)
        return penalty

    return jax.vmap(sample_penalty)

# def make_reverse_penalty_cost():
#     """
#     Penalizza solo la retromarcia (v < -0.05) con penalità quadratica continua.
#     Nessuna penalità per v >= -0.05
#     """
#     def sample_penalty(U_seq):
#         v = U_seq[:, :, 1]  # velocità lineare, shape (T, n)
#         threshold = -0.05
#         diff = v - threshold  # sarà negativo se v < -0.05
#         penalty = jnp.where(v < threshold, diff ** 2, 0.0)
#         return jnp.mean(penalty)

#     return jax.vmap(sample_penalty)

def make_goal_tracking_cost(xg, x0):
    """
    Crea una funzione che calcola il costo medio normalizzato di tracking del goal
    per ogni traiettoria (batch di traiettorie locali).

    Args:
        xg: obiettivi dei robot, shape (n, 3)
        x0: posizioni iniziali dei robot, shape (n, 3)

    Returns:
        goal_cost_fn: funzione (trajs_window) → (N,)
    """
    goal_pos = xg[:, :2]  
    start_pos = x0[:, :2] 
    goal_dists = jnp.linalg.norm(start_pos - goal_pos, axis=1) + 1e-6  # per evitare div zero
   
    def sample_cost(traj):  
        pos = traj[:, :, :2]  
        diff = pos - goal_pos  
        dists = jnp.linalg.norm(diff, axis=-1)  
        norm_dists = dists / goal_dists  
        return jnp.mean(norm_dists)

    return jax.vmap(sample_cost)
 
# === COSTO DI COLLISIONE ===
def make_log_barrier_collision_cost(obs,n, Ra, epsilon=1e-6):
    """
    Costo continuo, sempre positivo vicino, zero lontano, senza soglia.
    """
    def cost_fn(trajs): 
        def sample_cost(traj):  
            def timestep_cost(state_t):  
                pos = state_t[:, :2]
                idx_i, idx_j = jnp.triu_indices(n, k=1)
                diffs = pos[idx_i] - pos[idx_j]
                dists = jnp.linalg.norm(diffs, axis=1)

                dist_safe = jnp.clip(dists - 2 * Ra, a_min=epsilon)
                if obs:
                    barrier = -jnp.log(dist_safe) # ostacoli no commento
                   
                else:
                    barrier = jnp.maximum(-jnp.log(dist_safe), 0.0)

                
                # scaled_dist = jnp.minimum(dists / 0.3, 1.0)
                # barrier = jnp.log(scaled_dist) / jnp.log(0.2 / 0.5)
                # penalty = -100*barrier
                return jnp.mean(barrier)

            return jnp.mean(jax.vmap(timestep_cost)(traj))

        return jax.vmap(sample_cost)(trajs)

    return cost_fn


def make_formation_cost_fn(x0_all):
    """
    Restituisce una funzione che calcola il costo di deformazione della formazione
    rispetto alla distanza iniziale tra i robot.
    """
    pos0 = x0_all[:, :2]
    diff0 = pos0[:, None, :] - pos0[None, :, :]
    dists0 = jnp.linalg.norm(diff0, axis=-1)
    mask = jnp.triu(jnp.ones((x0_all.shape[0], x0_all.shape[0]), dtype=bool), k=1)

    def formation_cost(trajs): 
        def sample_cost(traj):  
            def timestep_cost(state_t):
                pos = state_t[:, :2]
                diff = pos[:, None, :] - pos[None, :, :]
                dists = jnp.linalg.norm(diff, axis=-1)
                return jnp.sum(((dists - dists0) ** 2) * mask)

            return jnp.mean(jax.vmap(timestep_cost)(traj))

        return jax.vmap(sample_cost)(trajs)

    return formation_cost

# Computes the residual (final position error) from the goal
def make_residual_fn(penalize,state_init, env, Nsample):
    """
    Creates a residual function that measures the final position error
    of each robot with respect to the goal positions.
    
    """
    @jax.jit
    def residual_fn(pipeline_states):
        s0 = state_init.pipeline_state
        s0_batched = jnp.repeat(s0[None, ...], Nsample, axis=0)
        trajs = jnp.concatenate([s0_batched[:, None, :, :], pipeline_states], axis=1)
        x_Ts = trajs[:, -1, :, :2]
        theta_Ts = trajs[:, -1, :, 2]     # (Nsample, n)
        goal_error = x_Ts - env.xg[:, :2]
        if penalize:
            # Errore sull'orientamento: differenza angolare corretta
            theta_g = env.xg[:, 2]  # (n,)
            theta_diff = jnp.arctan2(jnp.sin(theta_Ts - theta_g), jnp.cos(theta_Ts - theta_g))  

            # Concatenazione completa: posizione (2) + orientamento (1)
            residual = jnp.concatenate([goal_error, theta_diff[:, :, None]], axis=-1)  # (Nsample, n, 3)
        else :
                residual = x_Ts - env.xg[:, :2]
        return  residual#goal_error

    return residual_fn


def make_static_obstacle_cost(static_obs, robot_radius, epsilon=1e-3, margin=0.05, scale=1.0):
    """
    Penalizza la vicinanza a ostacoli statici con log-barrier + 1/x, protetta da clip numerico.
    """
    def cost_fn(trajs):  
        def sample_cost(traj):  
            def timestep_cost(state_t):  
                x = state_t[:, 0]
                y = state_t[:, 1]
                cost = 0.0

                for x_c, y_c, w, h in static_obs:
                    obs_x_min = x_c - w / 2
                    obs_x_max = x_c + w / 2
                    obs_y_min = y_c - h / 2
                    obs_y_max = y_c + h / 2

                    dx = jnp.maximum(jnp.maximum(obs_x_min - x, 0.0), x - obs_x_max)
                    dy = jnp.maximum(jnp.maximum(obs_y_min - y, 0.0), y - obs_y_max)
                    dist = jnp.sqrt(dx**2 + dy**2 + epsilon)
                    x_clipped = jnp.minimum(dist/0.5, 1.0)
                    barrier = jnp.log(x_clipped) / jnp.log(2*robot_radius/0.5)

                    penalty = 100*barrier

                    cost += jnp.mean(penalty)

                return cost

            return jnp.mean(jax.vmap(timestep_cost)(traj))

        return jax.vmap(sample_cost)(trajs)

    return cost_fn

def make_orient_final_cost_fn(xg, w_theta=1.0, decay=10.0):
    """
    Crea una funzione di costo per l’orientamento finale coerente con gli altri termini:
    restituisce la media sui robot e batch.
    """
    goal_pos = xg[:, :2]
    theta_g = xg[:, 2]

    def sample_cost(traj):
        # traj: [T+1, n, 3]
        pos_T = traj[-1, :, :2]       # [n, 2]
        theta_T = traj[-1, :, 2]      # [n]

        dists = jnp.linalg.norm(pos_T - goal_pos, axis=-1)         # [n]
        weights = jnp.exp(-decay * dists)                          # [n]
        theta_diff = jnp.arctan2(jnp.sin(theta_T - theta_g),
                                 jnp.cos(theta_T - theta_g))       # [n]
        penalties = weights * theta_diff**2                        # [n]
        return w_theta * jnp.mean(penalties)                       # media sui robot

    return jax.vmap(sample_cost)  



# Lagrangian function for equality-constrained optimization
def make_lagrangian_fn(state_init, env, Nsample):
    residual_fn = make_residual_fn(env.penalize_backward,state_init, env, Nsample)
    log_barrier_fn = make_log_barrier_collision_cost(env.obstacles_enabled,env.n,env.Ra, epsilon=1e-3)
    goal_cost_fn = make_goal_tracking_cost(env.xg, env.x0)
    formation_cost_fn = make_formation_cost_fn(env.x0) if env.formation_shift else lambda x: 0.0
    obstacle_cost_fn = make_static_obstacle_cost(env.static_obstacles, env.Ra) if env.obstacles_enabled else lambda x: 0.0
    reverse_penalty_fn = make_reverse_penalty_cost() if env.penalize_backward else lambda x: 0.0
    orient_cost_fn = make_orient_final_cost_fn(env.xg) if env.penalize_backward else lambda x: 0.0


    @jax.jit
    def lagrangian(Y0s_windows, Y0s, pipeline_states,pipeline_states_window, lambda_goal, mu_k):
        s0 = state_init.pipeline_state
        s0_batched = jnp.repeat(s0[None, ...], Y0s.shape[0], axis=0) 
        trajs = jnp.concatenate([s0_batched[:, None, :, :], pipeline_states], axis=1) 
        
        control_cost_global = cost_fn(Y0s)
        control_cost_local =  cost_fn(Y0s_windows)

        reverse_penalty_global = reverse_penalty_fn(Y0s)
        reverse_penalty_local = reverse_penalty_fn(Y0s_windows)


        barrier_cost_global =log_barrier_fn(trajs)
        barrier_cost_local = log_barrier_fn(pipeline_states_window)
        
        goal_cost_global= goal_cost_fn(trajs)
        goal_cost_local= goal_cost_fn(pipeline_states_window)
        
        formation_cost_global = formation_cost_fn(trajs)
        formation_cost_local = formation_cost_fn(pipeline_states_window)

        obstacle_cost_global = obstacle_cost_fn(trajs)
        obstacle_cost_local = obstacle_cost_fn(pipeline_states_window)
       
        orient_cost_global = orient_cost_fn(trajs)
        orient_cost_local = orient_cost_fn(pipeline_states_window)


        h = residual_fn(pipeline_states)
        h_flat = h.reshape((Y0s.shape[0], -1))
        L_cost_local = control_cost_local +  30*barrier_cost_local  + 20* goal_cost_local+15*formation_cost_local+ 10*obstacle_cost_local #+orient_cost_local+ reverse_penalty_local
        # no ostacoli
        #L_cost_global = 0.5* control_cost_global + 25*barrier_cost_global + 30* goal_cost_global+15*formation_cost_global+  10*obstacle_cost_global #+ reverse_penalty_global + 30*orient_cost_global # NO OSTACOLI
        # OSTACOLI 
        L_cost_global = 0.5* control_cost_global + 25*barrier_cost_global + 30* goal_cost_global+15*formation_cost_global+  10*obstacle_cost_global+ 30*orient_cost_global+ 10*reverse_penalty_global
        L_constraint = jnp.dot(h_flat, lambda_goal) + 0.5 * mu_k * jnp.sum(h_flat ** 2, axis=1)
        L_tot = L_cost_global+L_constraint
        return L_cost_local,L_constraint,L_tot,control_cost_global,barrier_cost_global,goal_cost_global,h_flat,obstacle_cost_global, reverse_penalty_global, orient_cost_global
    return lagrangian