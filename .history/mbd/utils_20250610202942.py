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


# # Quadratic cost function on U
# @jax.jit
# def cost_fn(U_seq):
#     """
#     Somma dei quadrati dei controlli su batch di sequenze U_seq
#     """
#     return jnp.sum(U_seq ** 2, axis=(1, 2, 3))

# def make_goal_tracking_cost(xg, x0):
#     """
#     Crea una funzione che calcola il costo medio normalizzato di tracking del goal
#     per ogni traiettoria (batch di traiettorie locali).

#     Args:
#         xg: obiettivi dei robot, shape (n, 3)
#         x0: posizioni iniziali dei robot, shape (n, 3)

#     Returns:
#         goal_cost_fn: funzione (trajs_window) → (N,)
#     """
#     goal_pos = xg[:, :2]   # shape (n, 2)
#     start_pos = x0[:, :2]  # shape (n, 2)
#     goal_dists = jnp.linalg.norm(start_pos - goal_pos, axis=1) + 1e-6  # per evitare div zero
#     def sample_cost(traj):  # (L, n, 3)
#         pos = traj[:, :, :2]  # (L, n, 2)
#         diff = pos - goal_pos  # broadcasting (L, n, 2)
#         dists = jnp.linalg.norm(diff, axis=-1)  # (L, n)
#         norm_dists = dists / goal_dists  # (L, n)
#         return jnp.mean(norm_dists)

#     return jax.vmap(sample_cost)
 


# # === COSTO DI COLLISIONE ===

# # def make_log_barrier_collision_cost(n, Ra, epsilon=1e-6):
# #     """
# #     Costo continuo, sempre positivo vicino, zero lontano, senza soglia.
# #     """
# #     def cost_fn(trajs): 
# #         def sample_cost(traj):  
# #             def timestep_cost(state_t):  
# #                 pos = state_t[:, :2]
# #                 idx_i, idx_j = jnp.triu_indices(n, k=1)
# #                 diffs = pos[idx_i] - pos[idx_j]
# #                 dists = jnp.linalg.norm(diffs, axis=1)

# #                 dist_safe = jnp.clip(dists - 2 * Ra, a_min=epsilon)
# #                 barrier = -jnp.log(dist_safe) 
# #                 return jnp.mean(barrier)

# #             return jnp.mean(jax.vmap(timestep_cost)(traj))

# #         return jax.vmap(sample_cost)(trajs)

# #     return cost_fn
# def make_log_barrier_collision_cost(n, Ra, epsilon=1e-6, margin=0.05):
#     """
#     Log-barrier continua e normalizzata: penalizza solo se distanza < 2*Ra + margin.

#     Restituisce valori ~0 se lontani, ~-1 se molto vicini.
#     """
#     def cost_fn(trajs): 
#         def sample_cost(traj):  
#             def timestep_cost(state_t):  
#                 pos = state_t[:, :2]
#                 idx_i, idx_j = jnp.triu_indices(n, k=1)
#                 diffs = pos[idx_i] - pos[idx_j]
#                 dists = jnp.linalg.norm(diffs, axis=1)

#                 # distanza di sicurezza desiderata
#                 safe_d = 2 * Ra + margin

#                 # log-barrier normalizzata: 0 se sicuri, -1 se critica
#                 normed = -jnp.log(jnp.clip((safe_d - dists) / margin, a_min=epsilon))
#                 penalty = jnp.where(dists < safe_d, normed, 0.0)
                
#                 return jnp.mean(penalty)

#             return jnp.mean(jax.vmap(timestep_cost)(traj))

#         return jax.vmap(sample_cost)(trajs)

#     return cost_fn


# def make_formation_cost_fn(x0_all):
#     """
#     Restituisce una funzione che calcola il costo di deformazione della formazione
#     rispetto alla distanza iniziale tra i robot.
#     """
#     pos0 = x0_all[:, :2]
#     diff0 = pos0[:, None, :] - pos0[None, :, :]
#     dists0 = jnp.linalg.norm(diff0, axis=-1)
#     mask = jnp.triu(jnp.ones((x0_all.shape[0], x0_all.shape[0]), dtype=bool), k=1)

#     def formation_cost(trajs):  # (Nsample, L, n, 3)
#         def sample_cost(traj):  # (L, n, 3)
#             def timestep_cost(state_t):
#                 pos = state_t[:, :2]
#                 diff = pos[:, None, :] - pos[None, :, :]
#                 dists = jnp.linalg.norm(diff, axis=-1)
#                 return jnp.sum(((dists - dists0) ** 2) * mask)

#             return jnp.mean(jax.vmap(timestep_cost)(traj))

#         return jax.vmap(sample_cost)(trajs)

#     return formation_cost





# # Computes the residual (final position error) from the goal
# def make_residual_fn(state_init, env, Nsample):
#     """
#     Creates a residual function that measures the final position error
#     of each robot with respect to the goal positions.
    
#     """
#     @jax.jit
#     def residual_fn(pipeline_states):
#         s0 = state_init.pipeline_state
#         s0_batched = jnp.repeat(s0[None, ...], Nsample, axis=0)
#         trajs = jnp.concatenate([s0_batched[:, None, :, :], pipeline_states], axis=1)
#         x_Ts = trajs[:, -1, :, :2]
#         goal_error = x_Ts - env.xg[:, :2]
#         return goal_error

#     return residual_fn

# def make_static_obstacle_cost(static_obs, robot_radius, epsilon=1e-3, margin=0.05, scale=1.0):
#     """
#     Penalizza la vicinanza a ostacoli statici con log-barrier + 1/x, protetta da clip numerico.
#     """
#     def cost_fn(trajs):  # (Nsample, L, n, 3)
#         def sample_cost(traj):  # (L, n, 3)
#             def timestep_cost(state_t):  # (n, 3)
#                 x = state_t[:, 0]
#                 y = state_t[:, 1]
#                 cost = 0.0

#                 for x_c, y_c, w, h in static_obs:
#                     obs_x_min = x_c - w / 2
#                     obs_x_max = x_c + w / 2
#                     obs_y_min = y_c - h / 2
#                     obs_y_max = y_c + h / 2

#                     dx = jnp.maximum(jnp.maximum(obs_x_min - x, 0.0), x - obs_x_max)
#                     dy = jnp.maximum(jnp.maximum(obs_y_min - y, 0.0), y - obs_y_max)
#                     dist = jnp.sqrt(dx**2 + dy**2 + epsilon)

#                     safe_dist = jnp.clip(dist - robot_radius - margin, a_min=1e-2)
#                     penalty = -jnp.log(safe_dist) + scale / safe_dist

#                     cost += jnp.mean(penalty)

#                 return cost

#             return jnp.mean(jax.vmap(timestep_cost)(traj))

#         return jax.vmap(sample_cost)(trajs)

#     return cost_fn






# # Lagrangian function for equality-constrained optimization
# def make_lagrangian_fn(state_init, env, Nsample):
#     residual_fn = make_residual_fn(state_init, env, Nsample)
#     log_barrier_fn = make_log_barrier_collision_cost(env.n,env.Ra, epsilon=1e-3)
#     goal_cost_fn = make_goal_tracking_cost(env.xg, env.x0)
#     formation_cost_fn = make_formation_cost_fn(env.x0) if env.formation_shift else lambda x: 0.0
#     obstacle_cost_fn = make_static_obstacle_cost(env.static_obstacles, env.Ra) if env.obstacles_enabled else lambda x: 0.0
#     @jax.jit
#     def lagrangian(Y0s_windows, Y0s, pipeline_states,pipeline_states_window, lambda_goal, mu_k):
#         s0 = state_init.pipeline_state
#         s0_batched = jnp.repeat(s0[None, ...], Y0s.shape[0], axis=0) 
#         trajs = jnp.concatenate([s0_batched[:, None, :, :], pipeline_states], axis=1) 
        
#         control_cost_global = cost_fn(Y0s)
#         control_cost_local =  cost_fn(Y0s_windows)

#         barrier_cost_global =log_barrier_fn(trajs)
#         barrier_cost_local = log_barrier_fn(pipeline_states_window)
        
#         goal_cost_global= goal_cost_fn(trajs)
#         goal_cost_local= goal_cost_fn(pipeline_states_window)
        
#         formation_cost_global = formation_cost_fn(trajs)
#         formation_cost_local = formation_cost_fn(pipeline_states_window)

#         obstacle_cost_global = obstacle_cost_fn(trajs)
#         obstacle_cost_local = obstacle_cost_fn(pipeline_states_window)

#         h = residual_fn(pipeline_states)
#         h_flat = h.reshape((Y0s.shape[0], -1))
#         L_cost_local = 0.1*control_cost_local +  2*barrier_cost_local  +  goal_cost_local+formation_cost_local+ obstacle_cost_local
#         L_cost_global = 0.1* control_cost_global + 2*barrier_cost_global +  goal_cost_global+formation_cost_global+  obstacle_cost_global
#         L_constraint = jnp.dot(h_flat, lambda_goal) + 0.5 * mu_k * jnp.sum(h_flat ** 2, axis=1)
#         L_tot = L_cost_global+L_constraint
#         return L_cost_local,L_constraint,L_tot,control_cost_local,barrier_cost_local,goal_cost_local,h_flat,obstacle_cost_local
#     return lagrangian


# Quadratic cost function on U
@jax.jit
def cost_fn(U_seq):
    """
    Somma dei quadrati dei controlli su batch di sequenze U_seq
    """
    return jnp.sum(U_seq ** 2, axis=(1, 2, 3))

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
    goal_pos = xg[:, :2]   # shape (n, 2)
    start_pos = x0[:, :2]  # shape (n, 2)
    goal_dists = jnp.linalg.norm(start_pos - goal_pos, axis=1) + 1e-6  # per evitare div zero
   
    def sample_cost(traj):  # (L, n, 3)
        pos = traj[:, :, :2]  # (L, n, 2)
        diff = pos - goal_pos  # broadcasting (L, n, 2)
        dists = jnp.linalg.norm(diff, axis=-1)  # (L, n)
        norm_dists = dists / goal_dists  # (L, n)
        return jnp.mean(norm_dists)

    return jax.vmap(sample_cost)
 


# === COSTO DI COLLISIONE ===

# def make_log_barrier_collision_cost(n, Ra, epsilon=1e-6):
#     """
#     Costo continuo, sempre positivo vicino, zero lontano, senza soglia.
#     """
#     def cost_fn(trajs): 
#         def sample_cost(traj):  
#             def timestep_cost(state_t):  
#                 pos = state_t[:, :2]
#                 idx_i, idx_j = jnp.triu_indices(n, k=1)
#                 diffs = pos[idx_i] - pos[idx_j]
#                 dists = jnp.linalg.norm(diffs, axis=1)

#                 dist_safe = jnp.clip(dists - 2 * Ra, a_min=epsilon)
#                 barrier = -jnp.log(dist_safe) 
#                 return jnp.mean(barrier)

#             return jnp.mean(jax.vmap(timestep_cost)(traj))

#         return jax.vmap(sample_cost)(trajs)

#     return cost_fn
def make_log_barrier_collision_cost(n, Ra, epsilon=1e-6):
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
                barrier = -jnp.log(dist_safe) 
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
def make_residual_fn(state_init, env, Nsample):
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
        goal_error = x_Ts - env.xg[:, :2]
        return goal_error

    return residual_fn

def make_static_obstacle_cost(static_obs, robot_radius, epsilon=1e-3, margin=0.05, scale=1.0):
    """
    Penalizza la vicinanza a ostacoli statici con log-barrier + 1/x, protetta da clip numerico.
    """
    def cost_fn(trajs):  # (Nsample, L, n, 3)
        def sample_cost(traj):  # (L, n, 3)
            def timestep_cost(state_t):  # (n, 3)
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

                    safe_dist = jnp.clip(dist - robot_radius - margin, a_min=1e-2)
                    penalty = -jnp.log(safe_dist) + scale / safe_dist

                    cost += jnp.mean(penalty)

                return cost

            return jnp.mean(jax.vmap(timestep_cost)(traj))

        return jax.vmap(sample_cost)(trajs)

    return cost_fn






# Lagrangian function for equality-constrained optimization
def make_lagrangian_fn(state_init, env, Nsample):
    residual_fn = make_residual_fn(state_init, env, Nsample)
    log_barrier_fn = make_log_barrier_collision_cost(env.n,env.Ra, epsilon=1e-3)
    goal_cost_fn = make_goal_tracking_cost(env.xg, env.x0)
    formation_cost_fn = make_formation_cost_fn(env.x0) if env.formation_shift else lambda x: 0.0
    obstacle_cost_fn = make_static_obstacle_cost(env.static_obstacles, env.Ra) if env.obstacles_enabled else lambda x: 0.0
    @jax.jit
    def lagrangian(Y0s_windows, Y0s, pipeline_states,pipeline_states_window, lambda_goal, mu_k):
        s0 = state_init.pipeline_state
        s0_batched = jnp.repeat(s0[None, ...], Y0s.shape[0], axis=0) 
        trajs = jnp.concatenate([s0_batched[:, None, :, :], pipeline_states], axis=1) 
        
        control_cost_global = cost_fn(Y0s)
        control_cost_local =  cost_fn(Y0s_windows)

        barrier_cost_global =log_barrier_fn(trajs)
        barrier_cost_local = log_barrier_fn(pipeline_states_window)
        
        goal_cost_global= goal_cost_fn(trajs)
        goal_cost_local= goal_cost_fn(pipeline_states_window)
        
        formation_cost_global = formation_cost_fn(trajs)
        formation_cost_local = formation_cost_fn(pipeline_states_window)

        obstacle_cost_global = obstacle_cost_fn(trajs)
        obstacle_cost_local = obstacle_cost_fn(pipeline_states_window)

        h = residual_fn(pipeline_states)
        h_flat = h.reshape((Y0s.shape[0], -1))
        L_cost_local = control_cost_local +  30*barrier_cost_local  + 25* goal_cost_local+15*formation_cost_local+ 10*obstacle_cost_local
        L_cost_global =  control_cost_global + 30*barrier_cost_global + 25* goal_cost_global+15*formation_cost_global+10*  obstacle_cost_global
        L_constraint = jnp.dot(h_flat, lambda_goal) + 0.5 * mu_k * jnp.sum(h_flat ** 2, axis=1)
        L_tot = L_cost_global+L_constraint
        return L_cost_local,L_constraint,L_tot,control_cost_local,barrier_cost_local,goal_cost_local,h_flat,obstacle_cost_local
    return lagrangian