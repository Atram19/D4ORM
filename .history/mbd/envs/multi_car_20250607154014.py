import jax
from jax import numpy as jnp
from flax import struct
from functools import partial
import matplotlib.pyplot as plt
import mbd
import matplotlib.cm as cm
from dataclasses import dataclass

# Define command-line arguments
@dataclass
class Args:
    seed: int = 0
    n_robots: int = 4
    Nsample: int = 2048         # number of samples
    Hsample: int = 100          # horizon
    Ndiffuse: int = 100         # number of diffusion steps
    temp_sample: float = 0.1    # temperature for sampling
    # per ibrido
    beta0: float = 5e-4         # initial noise
    betaT: float = 2e-2         # final noise  # PER ECD DEVE ESSERE A 1 SE NO NON VAA
    # PER D4orm e ECD
    # beta0: float = 1e-4         # initial noise
    # betaT: float = 1e-2         # final noise  # PER ECD DEVE ESSERE A 1 SE NO NON VAA

    initial_sigma: float = 1   # initial gaussian noise
    alpha: float = 0.01         # optimization step size
    mu: float = 50             # penalty term
    noise_decay: float = 0.3    # decay factor for noise
    not_render: bool = False
    high_resolution: bool = False
    ECD : bool = False
    formation_shift: bool = False 
    T: int = 30
    save_video: bool = False 
    obstacles_enabled: bool = False  # Attiva penalità da ostacoli se True


def car_dynamics(x, u):
    # x = x.at[3].set(jnp.clip(x[3], -2.0, 2.0))
    return jnp.array(
        [
            u[1] * jnp.cos(x[2])*3.0,  # x_dot  # CAMBIO DINAMICA
            u[1] * jnp.sin(x[2])*3.0,  # y_dot
            u[0] * jnp.pi / 3*2 ,  # theta_dot
            # u[1] * 6.0,  # v_dot
        ]
    )

# Numerical integration using Runge-Kutta 4 

def rk4(dynamics, x, u, dt):
    k1 = dynamics(x, u)
    k2 = dynamics(x + dt / 2 * k1, u)
    k3 = dynamics(x + dt / 2 * k2, u)
    k4 = dynamics(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

# Check for inter-robot collisions

def check_inter_robot_collisions(X_t, Ra):
   
    pos = X_t[:, :2] 
    dists = jnp.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)               # Compute the pairwise distance matrix between all robots using broadcasting.
    collision_matrix = dists < 2 * Ra                                                 # Create a collision matrix: True if distance < 2 * Ra, False otherwise.
    collision_matrix = collision_matrix.at[jnp.diag_indices(pos.shape[0])].set(False) # Ignore self-collisions by setting diagonal to False.
    return bool(jnp.any(collision_matrix))

# Generate initial and goal positions for n robots arranged in antipodal pairs on a circle.

def antipodal_positions(n, radius):
    angles = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
    x0_xy = jnp.stack([
        radius * jnp.cos(angles),
        radius * jnp.sin(angles)
    ], axis=1)
    xg_xy = -x0_xy
    #theta0 = jnp.zeros(n) 
    # theta0 = jnp.ones(n) * (jnp.pi / 2)

    # theta_g = jnp.zeros(n)  
     # Calcolo dell'orientamento iniziale: angolo tra x0 e xg
    delta = xg_xy - x0_xy
    theta0 = jnp.arctan2(delta[:, 1], delta[:, 0])  # orientati verso il goal

    # Orientamento finale (opzionale): verso il centro o verso -x0?
    theta_g = jnp.arctan2(delta[:, 1], delta[:, 0])  
    
    x0 = jnp.hstack([x0_xy, theta0[:, None]])
    xg = jnp.hstack([xg_xy, theta_g[:, None]])
    # print(f"Initial states:\n{x0}\nFinal states:\n{xg}")
    return x0, xg

#  Shifts the entire robot formation while maintaining relative positions on the circle

def circular_shift_goals(n,radius, shift=(0.0, 3.0)):
        
        angles = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        C_start = jnp.array([0.0, 0.0])
        C_goal = C_start + jnp.array(shift)

        x0_xy = C_start + radius * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
        xg_xy = C_goal + radius * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
        directions = xg_xy - x0_xy
        theta0 = jnp.arctan2(directions[:, 1], directions[:, 0])

        x0 = jnp.hstack([x0_xy, theta0[:, None]])
        xg = jnp.hstack([xg_xy, theta0[:, None]])

        return x0, xg
        
def check_collision_static(pos, obstacles, Ra):
    def single_obs_check(obs):
        xc, yc, w, h = obs
        x_min = xc - w / 2
        x_max = xc + w / 2
        y_min = yc - h / 2
        y_max = yc + h / 2
        dx = jnp.maximum(jnp.maximum(x_min - pos[0], 0), pos[0] - x_max)
        dy = jnp.maximum(jnp.maximum(y_min - pos[1], 0), pos[1] - y_max)
        dist = jnp.sqrt(dx ** 2 + dy ** 2 + 1e-6)
        return dist < Ra

    return jnp.any(jax.vmap(single_obs_check)(obstacles))

def compute_r_safe(p, q_all, k, Ra, margin=0.05):
    others = q_all[:, :2]
    dists = jnp.linalg.norm(p - others, axis=1)
    is_other = jnp.arange(q_all.shape[0]) != k

    safe_d = 2 * Ra + margin
    # penalità log-barrier-like
    values = jnp.clip((dists - safe_d) / safe_d, 0.0, 1.0)
    masked = jnp.where(is_other, values, 0.0)
    return jnp.mean(masked)




@struct.dataclass
class State:
    pipeline_state: jnp.ndarray  # current state of the robot (x, y, theta)
    obs: jnp.ndarray             # optional : fixed obstacles 
    reward: jnp.ndarray          # reward for each robot 
    done: jnp.ndarray            # optional : goal reached 

class MultiCar2d:
    def __init__(self, n, radius=2.0, robot_radius=0.1,formation_shift = False,obstacles_enabled=False,ECD=False):
        self.n = n              # number of robots
        self.dt = 0.1           # time step
        self.H = 100            # horizon
        self.Ra = robot_radius  # radius of the robot
        self.radius = radius   # radius of the initial circle
        self.wt = 2             # weight for the reward function
        self.formation_shift = formation_shift
        self.obstacles_enabled = obstacles_enabled
        self. ECD = ECD
        if obstacles_enabled:
            # self.static_obstacles = jnp.array([
            #     [0.0, 1.5, 0.6, 0.1],
            #     [0.0, -1.5, 0.6, 0.1],
            #     [-1.5, 0.0, 0.1, 0.6],
            #     [1.5, 0.0, 0.1, 0.6],
            # ])
            
            # 
            self.static_obstacles = jnp.array([
                [0.0,  0.8, 1.8, 0.07],   # parete orizzontale in alto
                [0.0, -0.8, 1.8, 0.07],   # parete orizzontale in basso
                [-0.8, 0.0, 0.07, 1.8],   # parete verticale a sinistra
                [ 0.8, 0.0, 0.07, 1.8],   # parete verticale a destra
            ])

        else:
            self.static_obstacles = jnp.zeros((0, 4))

        
        if self.formation_shift:
            self.x0,self.xg = circular_shift_goals(n,radius = self.radius, shift=(0, 3.0))
        else:
            self.x0, self.xg = antipodal_positions(n, radius=self.radius)
    
    # reset the state of the environment to an initial state
    def reset(self, rng):
            return State(
                pipeline_state=self.x0,  
                obs=self.x0,             
                reward=jnp.zeros((self.n,)),  
                done=jnp.zeros((self.n,)),
    
            )
    
    # execute one time step within the environment
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: jax.Array) -> State:
        """
        Update the state of the environment based on the action taken by the agent.
        """
        action = jnp.clip(action, -1.0, 1.0)         
        q = state.pipeline_state                      

        # Compute the next state using Runge-Kutta 4
        q_new = jax.vmap(rk4, in_axes=(None, 0, 0, None))(
            car_dynamics, q, action, self.dt
        ) 

     

        # Compute reward su posizione aggiornata
        reward = self.get_rewards(q_new, action)


        return state.replace(pipeline_state=q_new, obs=q_new, reward=reward, done=jnp.zeros((self.n,)))


    # Calculate the reward 
    @partial(jax.jit, static_argnums=(0,))
    def get_rewards(self, q_all, u_all):
        """
        Compute the reward for each robot based on its current state, actions and the states of all robots.
        """

        def single_reward(k, q,u_all):
            u_k = u_all[k]  # action of the k-th robot
            p = q[:2]
            pT = self.xg[k][:2]
            p0 = self.x0[k][:2] 
            r_goal = 1.0 - jnp.linalg.norm(p - pT) / jnp.linalg.norm(p0 - pT)
            dist = jnp.linalg.norm(p - pT)
            #r_goal = jnp.exp(-2.5 * dist)
            dist = jnp.linalg.norm(p - pT)
            r_goal = -jnp.log(dist + 1e-3)
            goal_reached = jnp.linalg.norm(p - pT) < 0.2  # es. threshold = 0.2
            r_goal = jnp.where(goal_reached, 100.0, 0.0)

            # # r_safe = self.wt*-1.0 * jnp.any((dists <= 2 * self.Ra + 1e-2) & (jnp.arange(self.n) != k)) # NON FUNZIONA PER OSTACOLI 
            # r_safe = compute_r_safe(q[:2], q_all, k, self.Ra) 
            # # self.wt deve essere =2 quando non ho gli ostacoli fissi, mentre pari a 1 quando ho gli ostacoli fissi e cosi r_safe esponenziale funziona in entrambi i casi 
            # # con ECD senza self.wt per r_safe continua
            if self.ECD or self.obstacles_enabled:
                    r_safe = compute_r_safe(q[:2], q_all, k, self.Ra) 
                   
            else:
                    dists = jnp.linalg.norm(p - q_all[:, :2], axis=1)
                    r_safe = self.wt*-1.0 * jnp.any((dists <= 2 * self.Ra + 1e-2) & (jnp.arange(self.n) != k)) # NON FUNZIONA PER OSTACOLI 
           
            def rews_formation(q_all, x0_all):
                pos = q_all[:, :2]
                pos0 = x0_all[:, :2]
                diff = pos[:, None, :] - pos[None, :, :]
                diff0 = pos0[:, None, :] - pos0[None, :, :]
                dists = jnp.linalg.norm(diff, axis=-1)
                dists0 = jnp.linalg.norm(diff0, axis=-1)
                mask = jnp.triu(jnp.ones((self.n, self.n), dtype=bool), k=1)
                return jnp.mean((dists - dists0) ** 2 * mask)

            # Penalità per ostacoli statici (tipo "porte")
            def single_obs_penalty(p, obstacle):
                x_c, y_c, w, h = obstacle
                obs_x_min = x_c - w / 2
                obs_x_max = x_c + w / 2
                obs_y_min = y_c - h / 2
                obs_y_max = y_c + h / 2

                # distanza “morbida” per penalizzazione continua
                dx = jnp.maximum(jnp.maximum(obs_x_min - p[0], 0.0), p[0] - obs_x_max)
                dy = jnp.maximum(jnp.maximum(obs_y_min - p[1], 0.0), p[1] - obs_y_max)
                dist = jnp.sqrt(dx**2 + dy**2 + 1e-3)
                soft_penalty = -1.0 / (1.0 + jnp.exp(40 * (dist - (self.Ra + 0.15))))

                # penalità “dura” se dentro ostacolo
                is_inside = (
                    (p[0] >= obs_x_min - self.Ra) & (p[0] <= obs_x_max + self.Ra) &
                    (p[1] >= obs_y_min - self.Ra) & (p[1] <= obs_y_max + self.Ra)
                )
                hard_penalty = jnp.where(is_inside, -50.0, 0.0)

                return soft_penalty + hard_penalty

            if self.obstacles_enabled:
                obstacles = jnp.array(self.static_obstacles)
                r_obs_vals = jax.vmap(lambda obs: single_obs_penalty(p, obs))(obstacles)
                r_obstacles = jnp.mean(r_obs_vals)  # ∈ [0, 1]
               
            else:
                r_obstacles = 0.0


            r_form = -rews_formation(q_all, self.x0) if self.formation_shift else 0.0

            r_control =  - jnp.sum(u_k ** 2)  # penalizzazione sul controllo

            return 5*r_goal +  r_safe + r_form + 0.1*r_control+2.5* r_obstacles

        return jax.vmap(single_reward, in_axes=(0, 0, 0))(jnp.arange(self.n), q_all, u_all)

    def get_reward_terms(self, q_all, u_all):
        def single_reward_terms(k, q, u_all):
            u_k = u_all[k]  # action of the k-th robot
            p = q[:2]
            pT = self.xg[k][:2]
            p0 = self.x0[k][:2] 
            r_goal = 1.0 - jnp.linalg.norm(p - pT) / jnp.linalg.norm(p0 - pT)
            dist = jnp.linalg.norm(p - pT)
            #r_goal = jnp.exp(-2.5 * dist)
            dist = jnp.linalg.norm(p - pT)
            r_goal = -jnp.log(dist + 1e-3)
            goal_reached = jnp.linalg.norm(p - pT) < 0.2  # es. threshold = 0.2
            r_goal = jnp.where(goal_reached, 100.0, 0.0)

            # # r_safe = self.wt*-1.0 * jnp.any((dists <= 2 * self.Ra + 1e-2) & (jnp.arange(self.n) != k)) # NON FUNZIONA PER OSTACOLI 
            # r_safe = compute_r_safe(q[:2], q_all, k, self.Ra) 
            # # self.wt deve essere =2 quando non ho gli ostacoli fissi, mentre pari a 1 quando ho gli ostacoli fissi e cosi r_safe esponenziale funziona in entrambi i casi 
            # # con ECD senza self.wt per r_safe continua
            if self.ECD or self.obstacles_enabled:
                    r_safe = compute_r_safe(q[:2], q_all, k, self.Ra) 
                   
            else:
                    dists = jnp.linalg.norm(p - q_all[:, :2], axis=1)
                    r_safe = self.wt*-1.0 * jnp.any((dists <= 2 * self.Ra + 1e-2) & (jnp.arange(self.n) != k)) # NON FUNZIONA PER OSTACOLI 
           
            def rews_formation(q_all, x0_all):
                pos = q_all[:, :2]
                pos0 = x0_all[:, :2]
                diff = pos[:, None, :] - pos[None, :, :]
                diff0 = pos0[:, None, :] - pos0[None, :, :]
                dists = jnp.linalg.norm(diff, axis=-1)
                dists0 = jnp.linalg.norm(diff0, axis=-1)
                mask = jnp.triu(jnp.ones((self.n, self.n), dtype=bool), k=1)
                return jnp.mean((dists - dists0) ** 2 * mask)

            # Penalità per ostacoli statici (tipo "porte")
            def single_obs_penalty(p, obstacle):
                x_c, y_c, w, h = obstacle
                obs_x_min = x_c - w / 2
                obs_x_max = x_c + w / 2
                obs_y_min = y_c - h / 2
                obs_y_max = y_c + h / 2

                # distanza “morbida” per penalizzazione continua
                dx = jnp.maximum(jnp.maximum(obs_x_min - p[0], 0.0), p[0] - obs_x_max)
                dy = jnp.maximum(jnp.maximum(obs_y_min - p[1], 0.0), p[1] - obs_y_max)
                dist = jnp.sqrt(dx**2 + dy**2 + 1e-3)
                soft_penalty = -1.0 / (1.0 + jnp.exp(40 * (dist - (self.Ra + 0.15))))

                # penalità “dura” se dentro ostacolo
                is_inside = (
                    (p[0] >= obs_x_min - self.Ra) & (p[0] <= obs_x_max + self.Ra) &
                    (p[1] >= obs_y_min - self.Ra) & (p[1] <= obs_y_max + self.Ra)
                )
                hard_penalty = jnp.where(is_inside, -50.0, 0.0)

                return soft_penalty + hard_penalty

            if self.obstacles_enabled:
                obstacles = jnp.array(self.static_obstacles)
                r_obs_vals = jax.vmap(lambda obs: single_obs_penalty(p, obs))(obstacles)
                r_obstacles = jnp.mean(r_obs_vals)  # ∈ [0, 1]
               
            else:
                r_obstacles = 0.0


            r_form = -rews_formation(q_all, self.x0) if self.formation_shift else 0.0

            r_control =  - jnp.sum(u_k ** 2)  # penalizzazione sul controllo

            r_total= 5*r_goal +  r_safe + r_form + 0.1*r_control+2.5* r_obstacles

            return jnp.array([r_goal, r_safe, r_form, r_control, r_obstacles, r_total])

        return jax.vmap(single_reward_terms, in_axes=(0, 0, 0))(jnp.arange(self.n), q_all, u_all)
    
    def get_reward_trajectory_terms(self, q_seq, u_seq):
        """
        Valuta i termini di reward per ogni tempo lungo la traiettoria (q_seq, u_seq).
        Output: (H, n, 6)
        """
        def reward_at_t(q_all, u_all):
            return self.get_reward_terms(q_all, u_all)  # (n, 6)

        return jax.vmap(reward_at_t)(q_seq, u_seq)


    # size of the action space
    @property
    def action_size(self):
        return 2
    
    # size of the observation space
    @property
    def observation_size(self):
        return 3
    
    # number of robots
    @property
    def num_robots(self):
        return self.n
    
    

    #  Function for visualizing the environment
    def render(self, ax, X: jnp.ndarray, goals: jnp.ndarray = None):
        
        n = X.shape[0]
        cmap = plt.get_cmap('tab20', n)
 
        for i in range(n):
            traj = X[i]             
            x, y, theta = traj[-1]  
            start_x, start_y = traj[0, 0], traj[0, 1] 

            # color = f"C{i % n}"
            color = cmap(i)

            # Draw the trajectory
            ax.plot(traj[:, 0], traj[:, 1], '-', color=color, label=f"Robot {i}")
            ax.plot(start_x, start_y, 's', color=color, markersize=4, label=f"Start {i}")
            ax.plot(x,y,'*',color=color, markersize=7, label=f"End {i}")

            # Draw the goal position
            if goals is not None:
                 gx, gy = goals[i, 0], goals[i, 1]
                 ax.plot(gx, gy, 'x', color=color, markersize=6)

            # Draw the robot orientation
            dx = 0.3 * jnp.cos(theta)
            dy = 0.3 * jnp.sin(theta)
            ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.15, fc=color, ec=color)
        
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)
        
        # Display the two circles
        if self.formation_shift:
                c0 = self.x0[:, :2].mean(axis=0)
                circle0 = plt.Circle((c0[0], c0[1]), self.radius, color='gray', linestyle='--', fill=False)
                ax.add_patch(circle0)
                cg = self.xg[:, :2].mean(axis=0)
                circleg = plt.Circle((cg[0], cg[1]), self.radius, color='black', linestyle='--', fill=False)
                ax.add_patch(circleg)
        for x_c, y_c, w, h in self.static_obstacles:
            rect = plt.Rectangle((x_c - w / 2, y_c - h / 2), w, h,
                                linewidth=1, edgecolor='red', facecolor='red', alpha=0.5)
            ax.add_patch(rect)

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
