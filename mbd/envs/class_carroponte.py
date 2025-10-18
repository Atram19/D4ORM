# class_crane.py
# Ambiente dinamico per il carroponte con pendolo (swing-up)
# Convenzione: θ = 0 in alto, θ = π in basso 

import jax
import jax.numpy as jnp
from flax import struct
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import os

# ===================== Args ===================== #
@struct.dataclass
class Args:
    seed: int = 42
    Nsample: int = 4096
    Hsample: int = 200
    Ndiffuse: int = 100
    beta0: float = 1e-3  #1e-4
    betaT: float = 1e-4  # 1e-2
    temp_sample: float = 0.1


# ===================== Stato ===================== #
@struct.dataclass
class State:
    pipeline_state: jnp.ndarray  # stato: [theta, dtheta, x, dx]
    reward: float
    r_terms: jnp.ndarray

# ===================== Rollout ===================== #
def rollout_single_us(step_env, state, us):
    """
    Esegue un rollout del sistema a partire da uno stato iniziale e
    da una sequenza di comandi us.
    step_env: funzione step dell'ambiente
    state   : stato iniziale (oggetto State)
    us      : array di azioni (T,)
    """
    def step_fn(state, u_t):
        state = step_env(state, u_t)
        return state, (state.reward, state.pipeline_state, state.r_terms)

    # esecuzione rollout
    _, (rews, states, r_terms) = jax.lax.scan(step_fn, state, us)

    # aggiungo lo stato iniziale all'inizio della traiettoria
    states = jnp.vstack([state.pipeline_state[None], states])
    rews = jnp.hstack([0.0, rews])

    return rews, states, r_terms


# ===================== Env ===================== #
class CranePendulumEnv:
    def __init__(self, dt=0.04):
        self.dt = dt
        self.H = 200

        # Parametri fisici (coerenti con do-mpc)
        self.M = 10.0   # massa carrello [kg]
        self.m = 1.0    # massa pendolo [kg]
        self.l = 1.0    # lunghezza pendolo [m]
        self.g = 9.81   # gravità [m/s^2]
        self.b = 0.0    # attrito

        # Azione massima (forza)
        self.max_u = 50.0

        # Stato iniziale (pendolo vicino al basso)
        # Convenzione: theta=π → basso
        self.q0 = jnp.array([0.9*jnp.pi, 0.0, 0.0, 0.0])
        # Stato obiettivo (pendolo in alto)
        self.qf = jnp.array([0.0, 0.0, -0.8, 0.0])

    # =============== Dinamica continua =============== #
    def dynamics(self, q, u):
        theta, dtheta, x, dx = q
        
        denom = self.M + self.m * jnp.sin(theta)**2

        theta_ddot = (
            self.M*self.g*jnp.sin(theta) + self.b*dx*jnp.cos(theta) + self.g*self.m*jnp.sin(theta)
            - 0.5*self.l*self.m*dtheta**2*jnp.sin(2*theta) - u*jnp.cos(theta)
        ) / (self.l * denom)

        x_ddot = (
            -self.b*dx - 0.5*self.g*self.m*jnp.sin(2*theta)
            + self.l*self.m*dtheta**2*jnp.sin(theta) + u
        ) / denom

        return  jnp.array([jnp.squeeze(dtheta), jnp.squeeze(theta_ddot), jnp.squeeze(dx), jnp.squeeze(x_ddot)])

    # =============== Integrazione RK4 =============== #
    def rk4(self, f, x, u, dt):
        k1 = f(x, u)
        k2 = f(x + dt/2 * k1, u)
        k3 = f(x + dt/2 * k2, u)
        k4 = f(x + dt * k3, u)
        return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    # =============== Reset e Step =============== #
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng):
        return State(pipeline_state=self.q0, reward=0.0, r_terms=jnp.zeros(3))

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: jax.Array) -> State:
        u = action * self.max_u
        q_next = self.rk4(self.dynamics, state.pipeline_state, u, self.dt)
        r, r_terms = self.get_reward(q_next, u)
        return State(pipeline_state=q_next, reward=r, r_terms=r_terms)
    

    def get_reward(self, q, u):
        theta, dtheta, x, dx = q

        # Errore angolare rispetto all’alto (theta = 0)
        angle_diff = jnp.arctan2(jnp.sin(self.qf[0]-theta), jnp.cos(self.qf[0]-theta))
        linear_error = self.qf[2]-x
        
        error_0 = jnp.pi
        linear_error_0 = self.qf[2]-self.q0[2]
        error_angle = jnp.sqrt(angle_diff**2)
        error_linear = jnp.sqrt(linear_error**2)
        r_goal = 0.8*(1-error_angle/error_0+1e-6)+0.2*(1-jnp.abs(error_linear)/(jnp.abs(linear_error_0)+1e-6))
        
        # r_carrello = 15*(1-jnp.abs(error_linear)/(jnp.abs(linear_error_0)+1e-6))
        # Penalità controllo
        #r_u = -0.1 * (jnp.abs(u)**2)

        
        E_kin = 0.5 * self.M * dx**2 + 0.5 * self.m * (
            (dx + self.l * dtheta * jnp.cos(theta))**2 + (self.l * dtheta * jnp.sin(theta))**2
        )
        E_pot = self.m * self.g * self.l * jnp.cos(theta)   
        # M_nom, m_nom, g, l = self.M, self.m, self.g, self.l
        # M_pert = M_nom *0.1+M_nom
        # m_pert = m_nom *0.1+m_nom

        # # === Energie con masse perturbate ===
        # E_kin = 0.5 * M_pert * dx**2 + 0.5 * m_pert * (
        #     (dx + l * dtheta * jnp.cos(theta))**2 + (l * dtheta * jnp.sin(theta))**2
        # )
        # E_pot = m_pert * g * l * ( jnp.cos(theta))


        r_total = +10*r_goal # -E_kin+10*E_pot
        r_total = jnp.squeeze(r_total)
        return r_total, jnp.array([jnp.squeeze(r_goal), jnp.squeeze(E_kin), jnp.squeeze(E_pot)])
    
    @property
    def action_size(self):
        return 1

    @property
    def observation_size(self):
        return 4
    
    def render(self, x_traj, U=None, rewards=None, r_terms=None, tag=""):
        """
        Visualizza gli stati, il controllo, le energie e la reward
        per il pendolo su carrello (convenzione θ=0 in alto).
        """
        x_traj = jax.device_get(x_traj)
        T = x_traj.shape[0]
        t = jnp.linspace(0, T * self.dt, T)

        theta = x_traj[:, 0]
        dtheta = x_traj[:, 1]
        x = x_traj[:, 2]
        dx = x_traj[:, 3]

        os.makedirs("results/crane/figs", exist_ok=True)

        # === 1. Stati principali ===
        fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
        fig.suptitle(f"Carroponte - Stati [{tag}]")

        axs[0].plot(t, theta)
        axs[0].set_ylabel("theta (rad)")

        axs[1].plot(t, dtheta)
        axs[1].set_ylabel("dtheta/dt (rad/s)")

        axs[2].plot(t, x)
        axs[2].set_ylabel("x (m)")

        axs[3].plot(t, dx)
        axs[3].set_ylabel("dx/dt (m/s)")
        axs[3].set_xlabel("Tempo (s)")

        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f"results/crane/figs/states_{tag}.png")
        plt.close()

        # === 2. Controllo ===
        if U is not None:
            U = jax.device_get(U).squeeze()
            t_u = jnp.linspace(0, T * self.dt, U.shape[0])

            plt.figure(figsize=(6, 3))
            plt.plot(t_u, 50*U)
            plt.title(f"Forza di controllo [{tag}]")
            plt.xlabel("Tempo (s)")
            plt.ylabel("F (N)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"results/crane/figs/control_{tag}.png")
            plt.close()

        # === 3. Energie e termini reward ===
        if r_terms is not None:
            r_terms = jax.device_get(r_terms)

            # Estrazione dei tre termini della reward
            r_goal = r_terms[:, 0]
            E_kin = r_terms[:, 1]
            E_pot = r_terms[:, 2]

            # === Grafico combinato ===
            fig, axs = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
            fig.suptitle(f"Termini della reward [{tag}]", fontsize=14)

            # 1. Tracking del goal
            axs[0].plot(t, r_goal, color='tab:green')
            axs[0].set_ylabel("r_goal")
            axs[0].set_title("Tracking del goal", fontsize=11)
            axs[0].grid(True)

            # 2. Penalità su velocità angolare
            axs[1].plot(t, E_kin, color='tab:orange')
            axs[1].set_ylabel("r_dtheta")
            axs[1].set_title("Energia cinetica", fontsize=11)
            axs[1].grid(True)

            # 3. Penalità sul controllo
            axs[2].plot(t, E_pot, color='tab:red')
            axs[2].set_ylabel("r_u")
            axs[2].set_title("Energia potenziale", fontsize=11)
            axs[2].set_xlabel("Tempo (s)")
            axs[2].grid(True)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(f"results/crane/figs/r_terms_{tag}.png")
            plt.close()
            # --- Grafico ---
            E_tot = E_kin + E_pot
            plt.figure(figsize=(8, 6))
            plt.plot(t, E_kin, label='Energia cinetica', color='tab:blue', linewidth=2)
            plt.plot(t, E_pot, label='Energia potenziale', color='tab:orange', linewidth=2)
            plt.plot(t, E_tot, label='Energia totale', color='tab:green', linestyle='--', linewidth=2)

            plt.title('Andamento delle energie nel tempo', fontsize=14)
            plt.xlabel('Tempo [s]', fontsize=12)
            plt.ylabel('Energia [J]', fontsize=12)
            plt.grid(True)
            plt.legend(fontsize=11)
            plt.tight_layout()
            plt.savefig(f"results/crane/figs/energies_{tag}.png", dpi=300)
            plt.close()


        # === 4. Reward totale ===
        if rewards is not None:
            rewards = jax.device_get(rewards)
            t_r = jnp.linspace(0, rewards.shape[0] * self.dt, rewards.shape[0])

            plt.figure(figsize=(6, 3))
            plt.plot(t_r, rewards)
            plt.title(f"Reward totale [{tag}]")
            plt.xlabel("Tempo (s)")
            plt.ylabel("Reward")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"results/crane/figs/reward_{tag}.png")
            plt.close()

        # === 5. Errore angolare rispetto all’equilibrio instabile (θ=0) ===
        delta_theta = jnp.mod(theta + jnp.pi, 2*jnp.pi) - jnp.pi
        plt.figure(figsize=(6, 3))
        plt.plot(t, delta_theta, label="Errore angolare (rad)")
        plt.axhline(0, color='k', linestyle='--', lw=1)
        plt.title(f"Errore rispetto all'equilibrio instabile (θ=0) [{tag}]")
        plt.xlabel("Tempo (s)")
        plt.ylabel("e_theta (rad)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/crane/figs/error_theta_{tag}.png")
        plt.close()
        

        
