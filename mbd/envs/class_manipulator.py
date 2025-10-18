import jax
import jax.numpy as jnp
from jax import jit
from dataclasses import dataclass
from functools import partial
from flax import struct
import mbd
from mbd.envs.manipulator import forward_kinematics_RRPR
# Importa qui le funzioni lambdificate da SymPy (generate separatamente)
# esempio placeholder, devi definire e importare queste funzioni
from mbd.envs.manipulator import B_func_jax, C_func_jax, G_func_jax
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from jax import debug
import matplotlib
import tkinter as tk
import tyro
from mpl_toolkits.mplot3d import Axes3D

def plot_four_panel(Y, t, titles, ylabels, suptitle, outfile):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4, 1, figsize=(10.5, 8), sharex=True)
    for i in range(4):
        axs[i].plot(t, Y[:, i], linewidth=1.5)
        axs[i].set_title(titles[i], fontsize=11)
        axs[i].set_ylabel(ylabels[i])
        axs[i].grid(True, alpha=0.3)
    axs[-1].set_xlabel("Step")
    fig.suptitle(suptitle, fontsize=13)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(f"{outfile}.png", dpi=300)
    plt.savefig(f"{outfile}.pdf")
    plt.close()

os.makedirs("results/manipulator", exist_ok=True)

matplotlib.use('Agg')
# Parametri fissi del robot
from brax.io import html
def dh_matrix(theta, d, a, alpha):
    ct, st = jnp.cos(theta), jnp.sin(theta)
    ca, sa = jnp.cos(alpha), jnp.sin(alpha)
    return jnp.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0]
    ])

@dataclass
class Args:
    seed: int = 42
    Nsample: int = 4096
    Hsample: int = 100
    Ndiffuse: int = 100
    beta0: float = 1e-3  #1e-2        # initial noise
    betaT: float = 1e-4  #1e-3        # final noise  
    temp_sample: float = 0.1
    save_video: bool = False
    Nsample_local = 64

@partial(jax.jit, static_argnums=(1)) 
def get_joint_positions(q, param):
    theta1, theta2, d3, theta4 = q
    a1, a2, a3, a4 = param[:, 0]
    alpha1, alpha2, alpha3, alpha4 = param[:, 1]
    d1, d2, _, d4 = param[:, 2]

    T01 = dh_matrix(theta1, d1, a1, alpha1)
    T12 = dh_matrix(theta2, d2, a2, alpha2)
    T23 = dh_matrix(0, d3, a3, alpha3)
    T34 = dh_matrix(theta4, d4, a4, alpha4)

    points = [np.zeros(3)]  # punto base
    T = np.eye(4)
    for T_next in [T01, T12, T23, T34]:
        T = T @ T_next
        point = np.array(T[:3, 3]).reshape(3,)
        points.append(point)


    return np.stack(points, axis=0)  # shape (5, 3)


@partial(jax.jit, static_argnums=(1,2,3,4,5))  # L1–D2 sono costanti
def forward_kinematics_rrpr_jax(q, L1, L2,L3,L4, D2):
    """
    q = [theta1, theta2, d3, theta4]
    DH params:
      Link1: a1=L1, alpha=0,     d1=0
      Link2: a2=L2, alpha=pi,    d2=0
      Link3: a3=0,  alpha=0,     d3=q[2]  (prismatic)
      Link4: a4=0,  alpha=0,     d4=D2
    """
    pi = jnp.pi
    theta1 = q[0]
    theta2 = q[1]
    theta4 = q[3]
    d3 = q[2]
    a = [L1, L2, L3, L4]
    alpha = [0, pi, 0, 0]
    d = [0, 0, d3, D2]


    T01 = dh_matrix(theta1, d[0], a[0], alpha[0])
    T12 = dh_matrix(theta2, d[1], a[1],alpha[1])
    T23 = dh_matrix(0.0, d[2],  a[2], alpha[2])
    T34 = dh_matrix(theta4, d[3], a[3], alpha[3])

    T04 = T01 @ T12 @ T23 @ T34
    return T04, T01, T12, T23, T34

def angle_diff(q, qf):
        return jnp.arctan2(jnp.sin(q - qf), jnp.cos(q - qf))


def rollout_single_us(step_env, state, us):
    def step_fn(state, u_t):
        state = step_env(state, u_t)
        return state, (state.reward, state.pipeline_state, state.r_terms)

    _, (rews, states, r_terms) = jax.lax.scan(step_fn, state, us)
    states = jnp.vstack([state.pipeline_state[None], states])
    rews = jnp.hstack([0.0, rews])
    return rews, states, r_terms


@struct.dataclass
class State:
    pipeline_state: jnp.ndarray  # stato: [q1,q2,q3,q4, dq1,dq2,dq3,dq4]
    reward: float
    r_terms: jnp.ndarray



def rk4(dynamics, x, u, dt):
    k1 = dynamics(x, u)
    k2 = dynamics(x + dt/2 * k1, u)
    k3 = dynamics(x + dt/2 * k2, u)
    k4 = dynamics(x + dt * k3, u)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def euler(dynamics, x, u, dt):
    return x + dt * dynamics(x, u)

@jax.jit
def B_func_jitted(q, m, L1, L2, L3, L4, D2):
    return B_func_jax(q, m, L1, L2, L3, L4, D2)

@jax.jit
def C_func_jitted(q, dq, m, L1, L2, L3, L4, D2):
    return C_func_jax(q, dq, m, L1, L2, L3, L4, D2)

@jax.jit
def G_func_jitted(q, m, g, L1, L2, L3, L4):
    return G_func_jax(q, m, g, L1, L2, L3, L4)

class RRPRSingleEnv:
    def __init__(self, dt=0.001):
        self.dt = dt
        self.H = 100
        self.ACTION_SCALE = jnp.array([175, 50, 75, 2])
        #self.ACTION_SCALE = jnp.array([20, 30, 25, 10])
        self.q0 = jnp.hstack([jnp.array([0.1, 0.1, 0.1, 0.1]), jnp.zeros(4)])
        self.qf = jnp.hstack([jnp.array ([-0.8,0.8,0.03,0.8]),jnp.zeros(4)])# stato iniziale (posizioni + velocità)
        #self.qf = jnp.hstack([jnp.array ([-2*jnp.pi,2*jnp.pi,0.03,2*jnp.pi]),jnp.zeros(4)])# stato iniziale (posizioni + velocità)
      
        self.L1_num = 0.40  # 40 cm → braccio principale
        self.L2_num = 0.30  # 30 cm → secondo braccio
        self.L3_num = 0.0   # prismatico
        self.D2_num = 0.10  # altezza (o offset) finale in z
        self.L4_num = self.D2_num

        self.m_num = jnp.array([
            6.0,   # m1: link lungo L1 (più robusto)
            4.0,   # m2: link lungo L2
            1.0,   # m3: prismatico (massa limitata)
            0.8    # m4: end-effector o giunto finale
        ])

        self.g0_num = jnp.array([0., 0., -9.81])
        # Parametri DH simbolici
        self.a = jnp.array([self.L1_num, self.L2_num, self.L3_num, self.L4_num])
        self.alpha = jnp.array([0, jnp.pi, 0, 0])
        self.d = jnp.array([0, 0, 0, self.D2_num])
        # Costruzione matrice parametri (4x3)
        self.param =jnp.array([self.a, self.alpha, self.d]).T  # shape (4, 3)
        self.q_min = jnp.array([-jnp.inf, -jnp.inf, 0.0, -jnp.inf]) 
        self.q_max = jnp.array([ jnp.inf,  jnp.inf, 0.3,  jnp.inf])



    def rrpr_dynamics(self,x, u):
        q = x[:4]
        dq = x[4:]
        
        B = B_func_jitted(q, self.m_num, self.L1_num, self.L2_num, self.L3_num, self.L4_num, self.D2_num)
        C = C_func_jitted(q, dq, self.m_num, self.L1_num, self.L2_num, self.L3_num, self.L4_num, self.D2_num)
        G = G_func_jitted(q, self.m_num, self.g0_num, self.L1_num, self.L2_num, self.L3_num, self.L4_num)

        G = G.squeeze()

        ddq = jnp.linalg.solve(B, u - C @ dq - G )
        s = jnp.linalg.svd(B, compute_uv=False)
        cond =s[0]/s[-1]
        #debug.print("B = {}", B)
        #debug.print("cond(B) = {}", cond)
        # debug.print("C @ dq = {}", C @ dq)
        # debug.print("G = {}", G)
        # debug.print("ddq = {}", ddq)

        return jnp.concatenate([dq, ddq])

    @partial(jax.jit, static_argnums=(0,))
    def reset(self,rng):
        return State(pipeline_state=self.q0, reward=0.0, r_terms=jnp.zeros(5))

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: jax.Array) -> State:
        action = jnp.clip(action, -1.0, 1.0)
        #action =  jnp.array([0.0, 0.0, -0.-0.23544, 0.0])

        action_scaled = action * self.ACTION_SCALE
        q_new = rk4(self.rrpr_dynamics, state.pipeline_state, action_scaled, self.dt)
    #    # Clipping sicuro dopo integrazione
        #q_new = q_new.at[:4].set(jnp.clip(q_new[:4], self.q_min, self.q_max))


        # Clamping solo sulle posizioni (prime 4 variabili)
        #q_new = euler(rrpr_dynamics, state.pipeline_state, action_scaled, self.dt)
        reward, r_terms = self.get_rewards(q_new, action_scaled)
        # jax.debug.print("r_terms = {}", r_terms)
        return State(pipeline_state=q_new, reward=reward,r_terms = r_terms)

    @partial(jax.jit, static_argnums=(0,))
    def get_rewards(self, q, u):
        pos = q[:4]
        vel = q[4:]

        # # === 1. Errore rispetto al goal in joint space
        # angle_errors_1 = angle_diff(self.q0[0],pos[0])
        # angle_errors_2 = angle_diff( self.q0[1],pos[1])
        # angle_errors_3 = angle_diff( self.q0[3],pos[3])
        # linear_error = self.qf[2] - pos[2]
        # # r_q_goal = -1.0 * (angle_errors_1**2 + angle_errors_2**2 + linear_error**2 + angle_errors_3**2)
        # joint_error = jnp.sqrt(
        #     angle_errors_1**2 +
        #     angle_errors_2**2 +
         
        #     angle_errors_3**2
        # )

        # r_q_goal = 7.0 * (1 - joint_error)+ 10.0 * (1 - jnp.abs(linear_error))
        # 1. Errore attuale
        angle_errors_1 = angle_diff(self.qf[0], pos[0])
        angle_errors_2 = angle_diff(self.qf[1], pos[1])
        angle_errors_3 = angle_diff(self.qf[3], pos[3])
        linear_error = self.qf[2] - pos[2]

        # 2. Errore iniziale (costante)
        angle_errors_1_0 = angle_diff(self.qf[0], self.q0[0])
        angle_errors_2_0 = angle_diff(self.qf[1], self.q0[1])
        angle_errors_3_0 = angle_diff(self.qf[3], self.q0[3])
        linear_error_0 = self.qf[2] - self.q0[2]

        # 3. Norme
        joint_error = jnp.sqrt(angle_errors_1**2 + angle_errors_2**2 + angle_errors_3**2)
        joint_error_0 = jnp.sqrt(angle_errors_1_0**2 + angle_errors_2_0**2 + angle_errors_3_0**2)

        # 4. Reward: cresce da 0 a 1
        r_q_goal = (
            0.5 * (1.0 - joint_error / (joint_error_0 + 1e-6)) +
            0.5 * (1.0 - jnp.abs(linear_error) / (jnp.abs(linear_error_0) + 1e-6))
        )

        
        # # === 2. Errore dell’end-effector (in workspace)
        # T_curr, *_ = forward_kinematics_rrpr_jax(pos, self.L1_num, self.L2_num, self.L3_num, self.L4_num, self.D2_num)
        # ee_pos = T_curr[:3, 3]

        # T_goal, *_ = forward_kinematics_rrpr_jax(self.qf[:4], self.L1_num, self.L2_num, self.L3_num, self.L4_num, self.D2_num)
        # ee_goal = T_goal[:3, 3]

        # err = jnp.linalg.norm( ee_goal-ee_pos)
        # r_goal = 5.0 * (1-err)
        # === 2. Errore dell’end-effector (in workspace) normalizzato allo start
        T_curr, *_ = forward_kinematics_rrpr_jax(pos, self.L1_num, self.L2_num, self.L3_num, self.L4_num, self.D2_num)
        ee_pos = T_curr[:3, 3]

        T_goal, *_ = forward_kinematics_rrpr_jax(self.qf[:4], self.L1_num, self.L2_num, self.L3_num, self.L4_num, self.D2_num)
        ee_goal = T_goal[:3, 3]

        # err corrente
        err = jnp.linalg.norm(ee_goal - ee_pos)

        # err0: distanza iniziale (calcolata dallo stato iniziale q0)
        T_start, *_ = forward_kinematics_rrpr_jax(self.q0[:4], self.L1_num, self.L2_num, self.L3_num, self.L4_num, self.D2_num)
        ee_start = T_start[:3, 3]
        err0 = jnp.maximum(jnp.linalg.norm(ee_goal - ee_start), 1e-8)  # evita divisioni per zero

        r_goal = 1.0 - (err / err0)
                

        # # === 3. Penalità sul controllo (solo vicino al goal)
        # r_control =  -0.01 * jnp.sum(abs(u)**2)

        # === 4. Penalità sulla velocità
        #r_vel = -1* jnp.sum(abs(vel)**2)
    
        
        # delta_u = u - u_required.squeeze()
        # r_delta = -1 * jnp.sum(delta_u**2)
        


        # vel_error = jnp.linalg.norm(self.qf[4:]-vel)
        # r_vel = 5.0 * (1-vel_error)
              

        # # === 6. Reward totale pesato
        # r_total = (
        #     #+ 1 * r_q_goal
        #      #+ 10.0 * r_goal
        #     # + 0.1 * r_control
        #      1.0 * r_vel
 
        # )
        tau_required = G_func_jitted(pos, self.m_num, self.g0_num, self.L1_num, self.L2_num, self.L3_num, self.L4_num)
        delta_tau = u - tau_required.squeeze()

        r_delta = -0.01 * jnp.sum(delta_tau ** 2)

        r_vel = -0.01 * jnp.sum(vel ** 2)
        #r_control = -0.01 * jnp.sum(u ** 2)
        # gains ∈ (0,1], massimi quando vel=0 e tau=tau_required
        k_vel  = 20.0
        k_tau  = 10
        gain_vel  = jnp.exp(-k_vel * jnp.sum(vel**2))
        gain_tau  = jnp.exp(-k_tau * jnp.sum(delta_tau**2))


        

        r_total = 10*r_q_goal+10* r_goal+ gain_tau +gain_vel  #+r_q_goal
        # bonus = jnp.where((err < 0.1) & (jnp.linalg.norm(vel) < 0.05), 0.5, 0.0)
        # r_total += bonus



        r_terms = jnp.array([r_goal, r_q_goal, gain_vel, gain_tau, r_total])
        
        return r_total, r_terms

      

    # size of the action space
    @property
    def action_size(self):
        return 4

    # size of the observation space
    @property
    def observation_size(self):
        return 8

    # number of robots
    @property
    def num_robots(self):
        return 1
    def render(self, X: jnp.ndarray, tau_seq: jnp.ndarray, rewards: jnp.ndarray = None, r_terms: jnp.ndarray = None, tag: str = ""):

        """
        Visualizzazione finale:
        - Salva animazione 3D del robot
        - Salva grafici statici per: posizioni angolari/prismatiche, velocità, torque, reward
        """
        

        os.makedirs("results/manipulator_diffusion", exist_ok=True)

        X = np.array(X)
        tau_seq = np.array(tau_seq)
        q_seq = X[:, :4]
        dq_seq = X[:, 4:]
        T = len(X)

        # === Indici giunti ===
        q_ang_idx = [0, 1, 3]  # Angolari
        q_pris_idx = [2]       # Prismatico

        # === POSIZIONI angolari ===
        # === NUOVA VISUALIZZAZIONE COMBINATA ===
        step = np.arange(T)

        titles_q   = ["q1", "q2", "q3", "q4 "]
        ylabels_q  = ["[rad]", "[rad]", "[m]", "[rad]"]
        plot_four_panel(
            Y=q_seq,
            t=step,
            titles=titles_q,
            ylabels=ylabels_q,
            suptitle="Joint positions",
            outfile=f"results/manipulator_diffusion/q_all_{tag}"
        )

        titles_dq  = ["dq1", "dq2", "dq3", "dq4"]
        ylabels_dq = ["[rad/s]", "[rad/s]", "[m/s]", "[rad/s]"]
        plot_four_panel(
            Y=dq_seq,
            t=step,
            titles=titles_dq,
            ylabels=ylabels_dq,
            suptitle="Joint velocities",
            outfile=f"results/manipulator_diffusion/dq_all_{tag}"
        )

        titles_tau  = [r"$\tau_1$ ", r"$\tau_2$ ", r"$\tau_3$ ", r"$\tau_4$"]
        ylabels_tau = ["[Nm]", "[Nm]", "[N]", "[Nm]"]
        plot_four_panel(
            Y=tau_seq,
            t=step,
            titles=titles_tau,
            ylabels=ylabels_tau,
            suptitle="Control torques/force",
            outfile=f"results/manipulator_diffusion/tau_all_{tag}"
        )

        # === TORQUE ===
        plt.figure(figsize=(12, 6))
        for i in range(4):
            plt.plot(tau_seq[:, i], label=f'tau{i+1}')
        plt.title("Azioni (Torque)")
        plt.xlabel("Step"); plt.ylabel("Tau")
        plt.grid(True); plt.legend()
        plt.savefig(f"results/manipulator_diffusion/azioni_{tag}.png")
        plt.close()

        # === REWARD ===
        if r_terms is not None:
            labels = ["r_goal", "r_q_goal", "r_vel", "r_tau"]
            ylabels = ["[-]"] * 4
            plot_four_panel(
                Y=r_terms[:, :4],
                t=step,
                titles=labels,
                ylabels=ylabels,
                suptitle="Reward components per step",
                outfile=f"results/manipulator_diffusion/reward_terms_{tag}"
            )

            # Plot reward totale a parte
            plt.figure(figsize=(10.5,3))
            plt.plot(step, r_terms[:, 4], linewidth=1.6)
            plt.xlabel("Step")
            plt.ylabel("[-]")
            plt.title("Total reward")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"results/manipulator_diffusion/reward_total_{tag}.png", dpi=300)
            plt.savefig(f"results/manipulator_diffusion/reward_total_{tag}.pdf")
            plt.close()


        # === TERMINI REWARD ===
        if r_terms is not None:
            labels = ["r_goal", "r_q_goal", "r_control", "r_vel", "r_total"]
            plt.figure(figsize=(10, 6))
            for i in range(min(r_terms.shape[1], 5)):
                plt.plot(r_terms[:, i], label=labels[i])
            plt.title("Termini della reward")
            plt.xlabel("Step"); plt.ylabel("Valore")
            plt.grid(True); plt.legend()
            plt.tight_layout()
            plt.savefig(f"results/manipulator_diffusion/reward_terms_{tag}.png")
            plt.close()
        # === Calcolo errore end-effector rispetto al goal nel tempo ===
        ee_errors = []
        for t in range(T):
            q = q_seq[t]
            T_curr, *_ = forward_kinematics_rrpr_jax(q, self.L1_num, self.L2_num, self.L3_num, self.L4_num, self.D2_num)
            ee_pos = np.array(T_curr[:3, 3])
            T_goal, *_ = forward_kinematics_rrpr_jax(self.qf[:4], self.L1_num, self.L2_num, self.L3_num, self.L4_num, self.D2_num)
            ee_goal = np.array(T_goal[:3, 3])
            err = np.linalg.norm(ee_goal - ee_pos)
            ee_errors.append(err)

        # === Plot errore nel tempo ===
        plt.figure(figsize=(10, 4))
        plt.plot(ee_errors, label="EE error to goal", color='tab:red')
        plt.xlabel("Step")
        plt.ylabel("Errore [m]")
        plt.title(f"Errore end-effector rispetto al goal ({tag})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/manipulator_diffusion/ee_error_{tag}.png", dpi=300)
        plt.close()

        # === ANIMAZIONE 3D ===
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-0.8, 0.8]); ax.set_ylim([-0.8, 0.8]); ax.set_zlim([-0.5, 0.3])
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.view_init(elev=45, azim=45)
        ax.grid(True)

        robot_line, = ax.plot([], [], [], 'ko-', linewidth=2)
        trail_line, = ax.plot([], [], [], 'b--', linewidth=1.5)
        goal_point, = ax.plot([], [], [], 'rx', markersize=8, label="Goal")
        title3d = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)
        ax.legend()

        trail_x, trail_y, trail_z = [], [], []

        def update(frame):
            nonlocal trail_x, trail_y, trail_z

            q = q_seq[frame]
            qf = np.array(self.qf[:4])

            # Forward kinematics
            T04, T01, T12, T23, T34 = forward_kinematics_rrpr_jax(q, self.L1_num, self.L2_num, self.L3_num, self.L4_num, self.D2_num)
            T02 = T01 @ T12
            T03 = T02 @ T23
            T04 = T03 @ T34

            p0 = jnp.array([0, 0, 0])
            p1 = T01[:3, 3]
            p2 = T02[:3, 3]
            p3 = T03[:3, 3]
            p4 = T04[:3, 3]
            points = np.stack([p0, p1, p2, p3, p4], axis=0)
            ee_pos = np.array(p4)

            # Goal
            T_goal, *_ = forward_kinematics_rrpr_jax(qf, self.L1_num, self.L2_num, self.L3_num, self.L4_num, self.D2_num)
            ee_goal = np.array(T_goal[:3, 3])

            robot_line.set_data(points[:, 0], points[:, 1])
            robot_line.set_3d_properties(points[:, 2])
            goal_point.set_data([ee_goal[0]], [ee_goal[1]])
            goal_point.set_3d_properties([ee_goal[2]])

            trail_x.append(ee_pos[0])
            trail_y.append(ee_pos[1])
            trail_z.append(ee_pos[2])
            trail_line.set_data(trail_x, trail_y)
            trail_line.set_3d_properties(trail_z)

            title3d.set_text(f"Frame {frame}")
            return robot_line, trail_line, goal_point, title3d

        ani = FuncAnimation(fig, update, frames=T, interval=50)
        ani.save(f"results/manipulator_diffusion/motion_3D_{tag}.mp4", writer='ffmpeg', fps=20)
        plt.close(fig)



def compute_kinetic_energy(q, dq,self):
    B = B_func_jax(q, self.m_num, self.L1_num, self.L2_num,self.L3_num,self.L4_num, self.D2_num)
   
    return 0.5 * dq @ B @ dq

def compute_potential_energy(q,self):
    T04, T01, T12, T23, T34= forward_kinematics_rrpr_jax(q, self.L1_num, self.L2_num,self.L3_num,self.L4_num, self.D2_num)

    T02 = T01 @ T12
    T03 = T02 @ T23
    T04 = T03 @ T34

    z1 = T01[2, 3]
    z2 = T02[2, 3]
    z3 = T03[2, 3]
    z4 = T04[2, 3]

    g = -self.g0_num[2]  

    V = self.m_num[0]*g*z1 + self.m_num[1]*g*z2 + self.m_num[2]*g*z3 + self.m_num[3]*g*z4
    return V

def compute_mechanical_energy(q, dq,self):
    T = compute_kinetic_energy(q, dq,env)
    V = compute_potential_energy(q,env)
    E = T+V
    return E,T,V






def simple_controller(state, qf,self):
    q = state[:4]
    dq = state[4:]
    qf = qf[:4]

    # Guadagni (aggiusta se serve)
    kp = jnp.array([100.0, 100.0, 500.0, 100.0])
    kd = jnp.array([20.0, 20.0, 100.0, 20.0])
    

    # Errore CAMBIARE ORDINEEEEEE
    e = jnp.array([
        angle_diff(qf[0],q[0]),   # rotazionale
        angle_diff( qf[1],q[1]),   # rotazionale
        qf[2]-q[2] ,              # PRISMATICO ← normale differenza
        angle_diff( qf[3],q[3])    # rotazionale
    ])

    de = -dq

    # Calcolo matrici dinamiche
    B = B_func_jax(q, self.m_num, self.L1_num, self.L2_num,self.L3_num,self.L4_num, self.D2_num)
    C = C_func_jax(q, dq, self.m_num, self.L1_num, self.L2_num,self.L3_num,self.L4_num,self.D2_num)
    G = G_func_jax(q, self.m_num, self.g0_num,self.L1_num,self.L2_num,self.L3_num,self.L4_num)
    G = G.squeeze()
    # Calcolo accelerazioni desiderate
    ddq_des = kp * e + kd * de

    # Formula torque = B*ddq_des + C*dq + G
    tau = B @ ddq_des + C @ dq + G

    return tau




# if __name__ == "__main__":

#     args = tyro.cli(Args)
#     env = RRPRSingleEnv(dt= 0.01)
#     step_env_jit = jax.jit(env.step)
#     reset_env_jit = jax.jit(env.reset)
#     state_init = reset_env_jit(jax.random.PRNGKey(args.seed))
    
#     state = reset_env_jit(jax.random.PRNGKey(0))  # Inizializza lo stato

    

#     rollout_us_fn = partial(rollout_single_us, step_env_jit, state_init)

#     positions = []
#     velocities = []
#     rewards = []
#     distances_per_joint = []
#     actions=[]
#     error =[]

   
#     for i in range(env.H):
#         action = simple_controller(state.pipeline_state, env.qf,env)  # nessuna coppia applicata
#         print(f"Step {i} — torque τ = {action}")
#         print(f"‣ max |τ| = {np.max(np.abs(action)):.2f}")

       
#         state = state = step_env_jit(state, action)

#         q = np.array(state.pipeline_state[:4])  # stato attuale
#         dq = np.array(state.pipeline_state[4:])  # velocità
#         # Errore rispetto al goal
#         err_0_1 = angle_diff(q[:2], env.qf[:2])         # angoli
#         err_2 = q[2] - env.qf[2]                         # prismatico
#         err_3 = angle_diff(q[3], env.qf[3])              # ultimo giunto rotazionale

#         err = np.array([err_0_1[0], err_0_1[1], err_2, err_3])  # errore per ogni giunto

#         positions.append(q)
#         velocities.append(dq)
#         rewards.append(state.reward)
#         distances_per_joint.append(np.abs(err))  # o err**2 se vuoi distanza quadratica
#         actions.append(action)
#         error.append(err)

#     positions = np.array(positions)  # shape (30,4)
#     velocities = np.array(velocities)  # shape (30,4)
#     rewards = np.array(rewards)  # shape (30,)
#     distances = np.array(distances_per_joint)
#     actions = np.array(actions)
#     errors  = np.array(error)
#      # Calcolo posizione finale end-effector
#     q_final = positions[-1]
#     param_final = np.array(env.param)
#     param_final[2, 2] = q_final[2]
#     points_final = get_joint_positions(env.qf, param_final)
#     ee_pos_final = points_final[-1]
#     print("\n=== Posizione finale end-effector ===")
#     print(f"X: {ee_pos_final[0]:.4f} m")
#     print(f"Y: {ee_pos_final[1]:.4f} m")
#     print(f"Z: {ee_pos_final[2]:.4f} m")
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.set_xlim([-0.8, 0.8])
#     ax.set_ylim([-0.8, 0.8])
#     ax.set_zlim([-0.5, 0.3])
#     ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
#     ax.view_init(elev=45, azim=45)
#     ax.grid(True)

#     robot_line, = ax.plot([], [], [], 'ko-', linewidth=2)
#     trail_line, = ax.plot([], [], [], 'r:', linewidth=1.5)
#     title = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

#     trail_x, trail_y, trail_z = [], [], []
#     goal_point, = ax.plot([], [], [], 'go', markersize=8, label='Goal')
#     # Plot posizioni
#     plt.figure(figsize=(12,6))
#     for i in range(4):
#         plt.plot(positions[:,i], label=f'pos q{i+1}')
#     plt.title('Evoluzione delle posizioni dei giunti')
#     plt.xlabel('Step')
#     plt.ylabel('Posizione')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig("results/manipulator/posizioni_giunti.png")
#     plt.show()
#     plt.figure(figsize=(12,6))
#     for i in range(4):
#         plt.plot(actions[:,i], label=f'pos q{i+1}')
#     plt.title('Evoluzione delle azioni ')
#     plt.xlabel('Step')
#     plt.ylabel('azioni')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig("results/manipulator/azioni.png")
#     plt.show()
#     # Plot velocità
#     plt.figure(figsize=(12,6))
#     for i in range(4):
#         plt.plot(velocities[:,i], label=f'vel dq{i+1}')
#     plt.title('Evoluzione delle velocità dei giunti')
#     plt.xlabel('Step')
#     plt.ylabel('Velocità')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig("results/manipulator/velocita_giunti.png")
#     plt.show()

#     # Plot reward
#     plt.figure(figsize=(8,4))
#     plt.plot(rewards)
#     plt.title('Reward ad ogni step')
#     plt.xlabel('Step')
#     plt.ylabel('Reward')
#     plt.grid(True)
#     plt.savefig("results/manipulator/reward.png")
#     plt.show()
    
#     plt.figure(figsize=(12,6))
#     for i in range(4):
#         plt.plot(errors[:,i], label=f' e{i+1}')
#     plt.title('errore per giunto')
#     plt.xlabel('Step')
#     plt.ylabel('errori')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig("results/manipulator/errore_giunti.png")
#     plt.show()

#     def update(frame):
#         qf = np.array(env.qf[:4])  # stato finale
#         q = np.array(positions[frame])  # converte da JAX a NumPy
#         param = np.array(env.param)

#         param[2, 2] = q[2]
#         points = get_joint_positions(q, param)  # (5, 3)
#         points_goal = get_joint_positions(qf, param)  # shape (5, 3)
#         ee_final = points[-1]
#         #x_goal = points_goal[-1]  # ultimo punto = end-effector
#         robot_line.set_data(points[:, 0], points[:, 1])
#         robot_line.set_3d_properties(points[:, 2])
#         qf = np.array(env.qf[:4])
#         param = np.array(env.param)
#         param[2, 2] = qf[2]  # imposta d3 = q3

#         T04, *_ = forward_kinematics_RRPR(param, qf, env.m_num)
#         x_goal = np.array(T04[:3, 3]).astype(np.float32).flatten()

#         trail_x.append(points[-1, 0])
#         trail_y.append(points[-1, 1])
#         trail_z.append(points[-1, 2])
#         trail_line.set_data(trail_x, trail_y)
#         trail_line.set_3d_properties(trail_z)
#         goal_point.set_data([x_goal[0]], [x_goal[1]])
#         goal_point.set_3d_properties([x_goal[2]])

#         error_xyz = ee_final - x_goal

#         # print("\n=== Errore finale end-effector ===")
#         # print(f"Errore X: {error_xyz[0]:.4f} m")
#         # print(f"Errore Y: {error_xyz[1]:.4f} m")
#         # print(f"Errore Z: {error_xyz[2]:.4f} m")


#         title.set_text(f"Frame {frame}")

#         return robot_line, trail_line, title,goal_point

#     ani = FuncAnimation(fig, update, frames=len(positions), interval=50, blit=False)
#     #plt.show()
      
#     ani.save("results/manipulator/manipulator_motion.mp4", writer='ffmpeg', fps=20)


if __name__ == "__main__":
    env = RRPRSingleEnv()
    state = env.reset(jax.random.PRNGKey(0))

    T = 150  # numero di step
    energy_total = []
    energy_kin = []
    energy_pot = []
    positions = []
    velocities = []
    for i in range(T):
        q = state.pipeline_state[:4]
        dq = state.pipeline_state[4:]
        E, T_kin, V_pot = compute_mechanical_energy(q, dq,env)
        energy_total.append(E)
        energy_kin.append(T_kin)
        energy_pot.append(V_pot)
        positions.append(np.array(q))
        velocities.append(np.array(dq))
        # Nessuna azione applicata
        action = jnp.zeros(4)
        state = env.step(state, action)

    energy_total = np.array(energy_total)
    energy_kin = np.array(energy_kin)
    energy_pot = np.array(energy_pot)
    positions = np.array(positions)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(energy_total, label='Energia Totale')
    plt.plot(energy_kin, label='Energia Cinetica')
    plt.plot(energy_pot, label='Energia Potenziale')
    plt.xlabel('Step')
    plt.ylabel('Energia [J]')
    plt.title('Verifica conservazione energia (azioni nulle)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/manipulator/energia_conservata.png")

    plt.show()

    print(f"ΔE_max = {np.max(energy_total) - np.min(energy_total):.6f} J")

    # === Animazione robot ===
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-0.8, 0.8])
    ax.set_ylim([-0.8, 0.8])
    ax.set_zlim([-0.8, 0.8])
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=45, azim=45)

    robot_line, = ax.plot([], [], [], 'ko-', lw=2)
    trail_line, = ax.plot([], [], [], 'r--', lw=1)
    title = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    trail_x, trail_y, trail_z = [], [], []

    def update(frame):
        q = positions[frame]
        param = np.array(env.param)
        param[2, 2] = q[2]  # imposta d3 prismatico
        points = get_joint_positions(q, param)  # (5, 3)

        robot_line.set_data(points[:, 0], points[:, 1])
        robot_line.set_3d_properties(points[:, 2])

        trail_x.append(points[-1, 0])
        trail_y.append(points[-1, 1])
        trail_z.append(points[-1, 2])
        trail_line.set_data(trail_x, trail_y)
        trail_line.set_3d_properties(trail_z)

        title.set_text(f"Step {frame}")
        return robot_line, trail_line, title

    ani = FuncAnimation(fig, update, frames=T, interval=30, blit=False)
    os.makedirs("results/manipulator", exist_ok=True)

    ani.save("results/manipulator/energia_robot.mp4", writer='ffmpeg', fps=30)

    plt.show()


  

