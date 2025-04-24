import jax
from jax import numpy as jnp
from flax import struct
from functools import partial
import matplotlib.pyplot as plt
import mbd

def car_dynamics(x, u):
    # x = x.at[3].set(jnp.clip(x[3], -2.0, 2.0))
    return jnp.array(
        [
            u[1] * jnp.sin(x[2])*3.0,  # x_dot
            u[1] * jnp.cos(x[2])*3.0,  # y_dot
            u[0] * jnp.pi / 3 * 2.0,  # theta_dot
            # u[1] * 6.0,  # v_dot
        ]
    )

def rk4(dynamics, x, u, dt):
    k1 = dynamics(x, u)
    k2 = dynamics(x + dt / 2 * k1, u)
    k3 = dynamics(x + dt / 2 * k2, u)
    k4 = dynamics(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

# collsioni tra robot
def check_inter_robot_collisions(X_t, Ra):
   
    pos = X_t[:, :2]  # prendi solo x,y
    dists = jnp.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1) # matrice delle distanze tra tutti i robot con broadcasting
    collision_matrix = dists < 2 * Ra # crea una matrice true/false delle collisioni
    collision_matrix = collision_matrix.at[jnp.diag_indices(pos.shape[0])].set(False)  # ignora self-self
    return jnp.any(collision_matrix)  # almeno una collisione tra robot

def antipodal_positions(n, radius=2.0):
    angles = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
    x0_xy = jnp.stack([
        radius * jnp.cos(angles),
        radius * jnp.sin(angles)
    ], axis=1)
    xg_xy = -x0_xy
    theta0 = jnp.zeros(n) 
    theta_g = jnp.zeros(n)  
    x0 = jnp.hstack([x0_xy, theta0[:, None]])
    xg = jnp.hstack([xg_xy, theta_g[:, None]])
    print(f"Stati iniziali:, {x0}\nStati finali: ,{xg}")
    return x0, xg

@struct.dataclass
class State:
    pipeline_state: jnp.ndarray  # stato corrente 
    obs: jnp.ndarray          
    reward: jnp.ndarray         # reward corrente
    done: jnp.ndarray           

class MultiCar2d:
    def __init__(self, n, radius=2.0, robot_radius=0.1):
        self.n = n
        self.dt = 0.1
        self.H = 100
        self.Ra = robot_radius # raggio del robot
        self.radius = radius # raggio circonferenza punti iniziali.
        self.wt = 2  # peso del reward di collisione


        self.x0, self.xg = antipodal_positions(n, radius=self.radius)


    def reset(self, rng):
            return State(
                pipeline_state=self.x0,  # (n, 3)
                obs=self.x0,             # opzionale per ostacoli fissi 
                reward=jnp.zeros((self.n,)),  # opzionale: vettore reward
                done=jnp.zeros((self.n,))     # opzionale: vettore done
            )
        
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: jax.Array) -> State:
            """
            Aggiorna lo stato di tutti i robot con una singola integrazione
            """
            action = jnp.clip(action, -1.0, 1.0)           # (n, 2)
            q = state.pipeline_state                      # (n, 3)
            
            # Applica rk4 a ciascun robot
            q_new = jax.vmap(rk4, in_axes=(None, 0, 0, None))(
                car_dynamics, q, action, self.dt
            )  # (n, 3)

            # Calcola reward
            reward = self.get_rewards(q_new)  # (n,)

            return state.replace(pipeline_state=q_new, obs=q_new, reward=reward, done=jnp.zeros((self.n,)))

    @partial(jax.jit, static_argnums=(0,))
    def get_rewards(self, q_all):
        """
        q_all: (n, 3)
        """
        def single_reward(k, q):
             p = q[:2]
             pT = self.xg[k][:2] # posizione finale
             p0 = self.x0[k][:2] # posizione iniziale 
             r_goal = 1.0 - jnp.linalg.norm(p - pT) / jnp.linalg.norm(p0 - pT)

             dists = jnp.linalg.norm(p - q_all[:, :2], axis=1)
             r_safe = -1.0 * jnp.any((dists <= 2 * self.Ra + 1e-2) & (jnp.arange(self.n) != k)) #r_safe = -1 se ho collisioni, 0 altrimenti
             return r_goal + self.wt * r_safe

        return jax.vmap(single_reward, in_axes=(0, 0))(jnp.arange(self.n), q_all)
    
    
    @property
    def action_size(self):
        return 2

    @property
    def observation_size(self):
        return 3
    @property
    def num_robots(self):
        return self.n
       
    def render(self, ax, X: jnp.ndarray, goals: jnp.ndarray = None):
        
        n = X.shape[0]
        for i in range(n):
            traj = X[i]             # (H+1, 3)
            x, y, theta = traj[-1]  # stato finale (ultima posizione)
            start_x, start_y = traj[0, 0], traj[0, 1] # stato iniziale 

            # Colore unico per ciascun robot
            color = f"C{i % 10}"

            # Traccia la traiettoria
            ax.plot(traj[:, 0], traj[:, 1], '-', color=color, label=f"Robot {i}")
            ax.plot(start_x, start_y, 's', color=color, markersize=4, label=f"Start {i}")
            ax.plot(x,y,'*',color=color, markersize=7, label=f"End {i}")

             # Goal (se fornito)
            if goals is not None:
                 gx, gy = goals[i, 0], goals[i, 1]
                 ax.plot(gx, gy, 'x', color=color, markersize=6)

            dx = 0.3 * jnp.cos(theta)
            dy = 0.3 * jnp.sin(theta)
            ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.15, fc=color, ec=color)

        ax.set_aspect('equal')
        ax.grid(True)

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)


    