import jax
import jax.numpy as jnp
from jax import jit
from dataclasses import dataclass
from functools import partial
from flax import struct
from mbd.envs.manipolator import forward_kinematics_RRPR,dh_transform
# Importa qui le funzioni lambdificate da SymPy (generate separatamente)
# esempio placeholder, devi definire e importare queste funzioni
from mbd.envs.manipolator import B_func_jax, C_func_jax, G_func_jax

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from jax import debug
# Parametri fissi del robot
from brax.io import html


jax.config.update("jax_debug_nans", True)
@dataclass
class Args:
    seed: int = 42
    Nsample: int = 2048
    Hsample: int = 100
    Ndiffuse: int = 100
    beta0: float = 1e-4
    betaT: float = 1e-3
    temp_sample: float = 0.2
    save_video: bool = False


def get_joint_positions(q, param):
    theta1, theta2, d3, theta4 = q
    a1, a2, a3, a4 = param[:, 0]
    alpha1, alpha2, alpha3, alpha4 = param[:, 1]
    d1, d2, _, d4 = param[:, 2]

    T01 = dh_transform(theta1, d1, a1, alpha1)
    T12 = dh_transform(theta2, d2, a2, alpha2)
    T23 = dh_transform(0, d3, a3, alpha3)
    T34 = dh_transform(theta4, d4, a4, alpha4)

    points = [np.zeros(3)]  # punto base
    T = np.eye(4)
    for T_next in [T01, T12, T23, T34]:
        T = T @ T_next
        point = np.array(T[:3, 3]).reshape(3,)
        points.append(point)


    return np.stack(points, axis=0)  # shape (5, 3)


def dh_matrix(theta, d, a, alpha):
    ct, st = jnp.cos(theta), jnp.sin(theta)
    ca, sa = jnp.cos(alpha), jnp.sin(alpha)
    return jnp.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0]
    ])

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
    def step(state, u_t):
        state = step_env(state, u_t)
        return state, (state.reward, state.pipeline_state, state.r_terms)

    _, (rews, pipeline_states,r_terms_seq) = jax.lax.scan(step, state, us)

    # Inserisci lo stato iniziale all'inizio
    pipeline_states = jnp.vstack([state.pipeline_state[None], pipeline_states])
    rews = jnp.hstack([0.0, rews])

    return rews, pipeline_states,r_terms_seq


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


class RRPRSingleEnv:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.H = 100
        self.ACTION_SCALE = jnp.array([8.0, 8.0, 600.0, 2])

        self.q0 = jnp.hstack([jnp.array([0.1, 0.1, 0.1, 0.1]), jnp.zeros(4)])
        self.qf = jnp.hstack([jnp.array ([-0.8,0.8,0.03,0.8]),jnp.zeros(4)])# stato iniziale (posizioni + velocità)
        self.m_num = jnp.array([10., 5., 10., 2.])
        self.L1_num = 0.10
        self.L2_num = 0.05
        self.L3_num = 0.0
        self.D2_num = 0.02
        self.L4_num = self.D2_num
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
        

        #D = jnp.diag(jnp.array([0,0,0, 10.0]))  # damping più forte
        B = B_func_jax(q, self.m_num,self.L1_num, self.L2_num,self.L3_num,self.L4_num, self.D2_num)
        C = C_func_jax(q, dq, self.m_num, self.L1_num, self.L2_num,self.L3_num,self.L4_num,self.D2_num)
        G = G_func_jax(q, self.m_num, self.g0_num,self.L1_num,self.L2_num,self.L3_num,self.L4_num)
        G = G.squeeze()

        ddq = jnp.linalg.solve(B, u - C @ dq - G )
        s = jnp.linalg.svd(B, compute_uv=False)
        cond =s[0]/s[-1]
        debug.print("B = {}", B)
        debug.print("C @ dq = {}", C @ dq)
        debug.print("G = {}", G)
        debug.print("ddq = {}", ddq)

        return jnp.concatenate([dq, ddq])

    @partial(jax.jit, static_argnums=(0,))
    def reset(self,rng):
        return State(pipeline_state=self.q0, reward=0.0, r_terms=jnp.zeros(6))

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: jax.Array) -> State:
        #action = jnp.clip(action, -1.0, 1.0)
        #action_scaled = action * self.ACTION_SCALE
        q_new = rk4(self.rrpr_dynamics, state.pipeline_state, action, self.dt)
    #    # Clipping sicuro dopo integrazione
        #q_new = q_new.at[:4].set(jnp.clip(q_new[:4], self.q_min, self.q_max))


        # Clamping solo sulle posizioni (prime 4 variabili)
        #q_new = euler(rrpr_dynamics, state.pipeline_state, action_scaled, self.dt)
        reward, r_terms = self.get_rewards(q_new, action)
        return State(pipeline_state=q_new, reward=reward,r_terms = r_terms)

    @partial(jax.jit, static_argnums=(0,))
    def get_rewards(self, q, u):
        pos = q[:4]
        vel = q[4:]
        #pos = jnp.clip(pos, self.q_min, self.q_max)

        # --- 1. Errore rispetto al goal (in q-space)
        angle_errors_1 = angle_diff(pos[0], self.qf[0])
        angle_errors_2 = angle_diff(pos[1], self.qf[1])
        angle_errors_3 = angle_diff(pos[ 3], self.qf[3])
        linear_error = pos[2] - self.qf[2]
        r_q_goal = -1*(angle_errors_1**2+ linear_error**2+angle_errors_2**2+angle_errors_3**2)
        r_q4 = -100.0 * angle_diff(pos[3], self.qf[3])**2

         # --- 12 Errore rispetto alle velocita desiderate (in q-space)
        err_vel_1 = vel[0]- self.qf[4]
        err_vel_2= vel[1]- self.qf[5]
        err_vel_3 = vel[2]- self.qf[6]
        err_vel_4 = vel[3] - self.qf[7]
        #r_vel = -1*(err_vel_1**2+ err_vel_2**2+err_vel_3**2+100*err_vel_4**2)

        # --- 2. Errore end-effector (in workspace)
        T_curr,T01, T12, T23, T34 = forward_kinematics_rrpr_jax(pos, self.L1_num, self.L2_num,self.L3_num,self.L4_num, self.D2_num)
        ee_pos = T_curr[:3, 3]

        T_goal,T01, T12, T23, T34 = forward_kinematics_rrpr_jax(self.qf[:4], self.L1_num, self.L2_num, self.L3_num,self.L4_num,self.D2_num)
        ee_goal = T_goal[:3, 3]

        # Errori separati
        err_x = ee_pos[0] - ee_goal[0]

        err_y = ee_pos[1] - ee_goal[1]
        err_z = ee_pos[2] - ee_goal[2]
        err = jnp.linalg.norm(ee_pos - ee_goal)
        # Penalizzazione quadratica, pesata
        r_goal = -100*( err_x**2 +  err_y**2 + 20.0 * err_z**2)  # z più importante



        # --- 3. Penalità sull’azione (control effort)
        # Penalizza control effort solo se sei vicino al goal
        r_control = jnp.where(err < 0.05, -0.1 * jnp.sum(u**2), 0.0)

        # --- 4. Penalità sulla velocità
        #r_vel = - jnp.mean(vel[3]**2)
        # Penalizza velocità solo se sei vicino al goal
        r_vel = jnp.where( err< 0.05, -5.0 * jnp.sum(vel**2), 0.0)
        # # --- 5. Penalità su posizioni fuori dai limiti (soft barrier)
        eps = 1e-4
        q_upper_margin = jnp.clip(self.q_max - pos, a_min=eps, a_max=10.0)
        q_lower_margin = jnp.maximum(pos - self.q_min, eps)
        r_safe = -10.0 * (jnp.sum(jnp.log(q_upper_margin)) + jnp.sum(jnp.log(q_lower_margin)))

        dq_max = jnp.array([5.0, 5.0, 2.0, 3.0])
        dq_upper_margin = jnp.maximum(dq_max - vel, eps)
        dq_lower_margin = jnp.maximum(vel + dq_max, eps)  # simmetrico (±dq_max)

        r_dq_safe = -5.0 * (jnp.sum(jnp.log(dq_upper_margin)) + jnp.sum(jnp.log(dq_lower_margin)))

        # --- 6. Reward finale (pesato)
        r_total = (
            + 5.0 * r_q_goal
            + 5.0 * r_goal
            + 0.001 * r_control
            + 0.01 * r_vel
            # + 1.0 * r_safe         # log-barrier su posizioni
            # +2*r_dq_safe
        )

        r_terms = jnp.array([r_goal, r_q_goal, r_control, r_vel, r_safe, r_total])

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
    def render(self, X: jnp.ndarray, tau_seq: jnp.ndarray,rewards: jnp.ndarray = None,r_terms : jnp.ndarray=None):
        """
        Anima la traiettoria del manipolatore RRPR in 3D
        + Mostra anche le q, dq, e torque nel tempo
        """
        X = np.array(X)
        tau_seq = np.array(tau_seq)
        q_seq = X[:, :4]
        dq_seq = X[:, 4:]

        T = len(X)

        fig = plt.figure(figsize=(14, 12))
        ax3d = fig.add_subplot(3, 2, 1, projection='3d')
        ax_q = fig.add_subplot(3, 2, 2)
        ax_dq = fig.add_subplot(3, 2, 3)
        ax_tau = fig.add_subplot(3, 2, 4)
        ax_r = fig.add_subplot(3, 2, 5)   # Reward
        ax_e = fig.add_subplot(3, 2, 6)   # Error


        fig2 = plt.figure()
        # 3D settings
        ax3d.set_xlim([-0.2, 0.2])
        ax3d.set_ylim([-0.2, 0.2])
        ax3d.set_zlim([-0.5, 0.3])
        ax3d.set_xlabel('x'); ax3d.set_ylabel('y'); ax3d.set_zlabel('z')
        ax3d.view_init(elev=45, azim=45)
        ax3d.grid(True)
        reward_line, = ax_r.plot([], [], 'g-', label="Reward")
        error_line, = ax_e.plot([], [], 'r-', label="Goal Error")

        ax_r.set_title("Reward over time"); ax_r.set_xlabel("t"); ax_r.set_ylabel("R(t)")
        ax_r.grid(True); ax_r.legend()

        ax_e.set_title("Goal error over time"); ax_e.set_xlabel("t"); ax_e.set_ylabel("||ee - goal||")
        ax_e.grid(True); ax_e.legend()


        robot_line, = ax3d.plot([], [], [], 'ko-', linewidth=2)
        trail_line, = ax3d.plot([], [], [], 'b--', linewidth=1.5)
        goal_point, = ax3d.plot([], [], [], 'rx', markersize=8, label='Goal')
        title3d = ax3d.text2D(0.05, 0.95, "", transform=ax3d.transAxes)
        ax3d.legend()

        trail_x, trail_y, trail_z = [], [], []
        error_buffer = []

        # Plot placeholders
        q_lines = [ax_q.plot([], [], label=f'q{i+1}')[0] for i in range(4)]
        dq_lines = [ax_dq.plot([], [], label=f'dq{i+1}')[0] for i in range(4)]
        tau_lines = [ax_tau.plot([], [], label=f'tau{i+1}')[0] for i in range(4)]

        for ax, name in [(ax_q, 'q'), (ax_dq, 'dq'), (ax_tau, 'tau')]:
            ax.set_xlim([0, T])
            ax.set_ylabel(name)
            ax.set_xlabel("Step")
            ax.grid(True)
            ax.legend()
        # == Se ci sono i r_terms, plot separato ==
        if r_terms is not None:
            fig_terms = plt.figure(figsize=(10, 6))
            ax_terms = fig_terms.add_subplot(1, 1, 1)
            labels = ["r_goal", "r_q_goal", "r_control", "r_vel", "r_safe", "r_total"]
            for i in range(min(r_terms.shape[1], 6)):
                ax_terms.plot(r_terms[:, i], label=labels[i])
            ax_terms.set_title("Andamento dei singoli termini di reward")
            ax_terms.set_xlabel("Step")
            ax_terms.set_ylabel("Valore")
            ax_terms.grid(True)
            ax_terms.legend()
            fig_terms.tight_layout()

        def update(frame):
            nonlocal trail_x, trail_y, trail_z

            if frame == 0:
                trail_x.clear()
                trail_y.clear()
                trail_z.clear()

            q = q_seq[frame]
            dq = dq_seq[frame]
            tau = tau_seq[frame]

            qf = np.array(self.qf[:4])
            L1, L2, D2 = self.L1_num, self.L2_num, self.D2_num

            # Calcola le trasformazioni intermedie
            T04, T01, T12, T23, T34 = forward_kinematics_rrpr_jax(q, self.L1_num, self.L2_num,self.L3_num,self.L4_num, self.D2_num)
            T02 = T01 @ T12
            T03 = T02 @ T23
            T04 = T03 @ T34

            # Joint positions: base, joint1, joint2, joint3, end-effector
            p0 = jnp.array([0, 0, 0])
            p1 = T01[:3, 3]
            p2 = T02[:3, 3]
            p3 = T03[:3, 3]
            p4 = T04[:3, 3]  # end-effector

            points = np.stack([p0, p1, p2, p3, p4], axis=0)
            ee_pos = np.array(p4)

            # Goal position (end-effector)
            T_goal = forward_kinematics_rrpr_jax(qf, self.L1_num, self.L2_num,self.L3_num,self.L4_num, self.D2_num)
            ee_goal = np.array(T_goal[:3, 3])
            e = ee_pos-ee_goal
            # Aggiorna linea robotica
            robot_line.set_data(points[:, 0], points[:, 1])
            robot_line.set_3d_properties(points[:, 2])
            goal_point.set_data([ee_goal[0]], [ee_goal[1]])
            goal_point.set_3d_properties([ee_goal[2]])

            # Trail
            trail_x.append(ee_pos[0])
            trail_y.append(ee_pos[1])
            trail_z.append(ee_pos[2])
            trail_line.set_data(trail_x, trail_y)
            trail_line.set_3d_properties(trail_z)
            title3d.set_text(f"Frame {frame}")

            # q, dq, tau plots
            x_vals = np.arange(frame + 1)
            for i in range(4):
                q_lines[i].set_data(x_vals, q_seq[:frame+1, i])
                dq_lines[i].set_data(x_vals, dq_seq[:frame+1, i])
                tau_lines[i].set_data(x_vals, tau_seq[:frame+1, i])
                for ax in [ax_q, ax_dq, ax_tau]:
                    ax.set_xlim([0, T])
                    ax.relim(); ax.autoscale_view()

            if rewards is not None:
                reward_line.set_data(np.arange(frame + 1), rewards[:frame+1])
                ax_r.set_xlim([0, T])
                ax_r.relim(); ax_r.autoscale_view()

            error_norm = np.linalg.norm(e)
            if frame == 0:
                self.error_buffer = [error_norm]
            else:
                self.error_buffer.append(error_norm)
            error_line.set_data(np.arange(frame + 1), self.error_buffer)
            ax_e.set_xlim([0, T])
            ax_e.relim()
            ax_e.autoscale_view()

            if frame == T - 1:
                fig.savefig("last_frame.png", dpi=300)

            return (
                robot_line, trail_line, title3d, goal_point,
                *q_lines, *dq_lines, *tau_lines,
                reward_line, error_line
            )
        
       

        ani = FuncAnimation(fig, update, frames=T, interval=50)
        ani.save("motion_extended.mp4", writer='ffmpeg', fps=20)
        plt.tight_layout()
        plt.show()


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

    g = -self.g0_num[2]  # gravità positiva = 9.81

    V = self.m_num[0]*g*z1 + self.m_num[1]*g*z2 + self.m_num[2]*g*z3 + self.m_num[3]*g*z4
    return V

def compute_mechanical_energy(q, dq):
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
        angle_diff(q[0], qf[0]),   # rotazionale
        angle_diff(q[1], qf[1]),   # rotazionale
        q[2] - qf[2],              # PRISMATICO ← normale differenza
        angle_diff(q[3], qf[3])    # rotazionale
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



if __name__ == "__main__":
    env = RRPRSingleEnv()
    state = env.reset(jax.random.PRNGKey(0))  # Inizializza lo stato

    positions = []
    velocities = []
    rewards = []
    distances_per_joint = []
    actions=[]
    error =[]
    for i in range(env.H):
        action = simple_controller(state.pipeline_state, env.qf,env)  # nessuna coppia applicata
        print(f"Step {i} — torque τ = {action}")
        print(f"‣ max |τ| = {np.max(np.abs(action)):.2f}")

       
        state = env.step(state, action)
        q = np.array(state.pipeline_state[:4])  # stato attuale
        dq = np.array(state.pipeline_state[4:])  # velocità
        # Errore rispetto al goal
        err_0_1 = angle_diff(q[:2], env.qf[:2])         # angoli
        err_2 = q[2] - env.qf[2]                         # prismatico
        err_3 = angle_diff(q[3], env.qf[3])              # ultimo giunto rotazionale

        err = np.array([err_0_1[0], err_0_1[1], err_2, err_3])  # errore per ogni giunto

        positions.append(q)
        velocities.append(dq)
        rewards.append(state.reward)
        distances_per_joint.append(np.abs(err))  # o err**2 se vuoi distanza quadratica
        actions.append(action)
        error.append(err)

    positions = np.array(positions)  # shape (30,4)
    velocities = np.array(velocities)  # shape (30,4)
    rewards = np.array(rewards)  # shape (30,)
    distances = np.array(distances_per_joint)
    actions = np.array(actions)
    errors  = np.array(error)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-0.2, 0.2])
    ax.set_ylim([-0.2, 0.2])
    ax.set_zlim([-0.5, 0.3])
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.view_init(elev=45, azim=45)
    ax.grid(True)

    robot_line, = ax.plot([], [], [], 'ko-', linewidth=2)
    trail_line, = ax.plot([], [], [], 'r:', linewidth=1.5)
    title = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    trail_x, trail_y, trail_z = [], [], []
    goal_point, = ax.plot([], [], [], 'go', markersize=8, label='Goal')

    def update(frame):
        qf = np.array(env.qf[:4])  # stato finale
        q = np.array(positions[frame])  # converte da JAX a NumPy
        param = np.array(env.param)

        param[2, 2] = q[2]
        points = get_joint_positions(q, param)  # (5, 3)
        points_goal = get_joint_positions(qf, param)  # shape (5, 3)
        ee_final = points[-1]
        #x_goal = points_goal[-1]  # ultimo punto = end-effector
        robot_line.set_data(points[:, 0], points[:, 1])
        robot_line.set_3d_properties(points[:, 2])
        qf = np.array(env.qf[:4])
        param = np.array(env.param)
        param[2, 2] = qf[2]  # imposta d3 = q3

        T04, *_ = forward_kinematics_RRPR(param, qf, env.m_num)
        x_goal = np.array(T04[:3, 3]).astype(np.float32).flatten()

        trail_x.append(points[-1, 0])
        trail_y.append(points[-1, 1])
        trail_z.append(points[-1, 2])
        trail_line.set_data(trail_x, trail_y)
        trail_line.set_3d_properties(trail_z)
        goal_point.set_data([x_goal[0]], [x_goal[1]])
        goal_point.set_3d_properties([x_goal[2]])

        error_xyz = ee_final - x_goal

        # print("\n=== Errore finale end-effector ===")
        # print(f"Errore X: {error_xyz[0]:.4f} m")
        # print(f"Errore Y: {error_xyz[1]:.4f} m")
        # print(f"Errore Z: {error_xyz[2]:.4f} m")


        title.set_text(f"Frame {frame}")

        return robot_line, trail_line, title,goal_point

    ani = FuncAnimation(fig, update, frames=len(positions), interval=50, blit=False)
    plt.show()
    #ani.save("manipulator_motion.mp4", writer='ffmpeg', fps=20)