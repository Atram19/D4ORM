import numpy as np

from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('Agg')  # prima di importare pyplot
import matplotlib.pyplot as plt
from mbd.envs.class_manipulator import RRPRSingleEnv, forward_kinematics_rrpr_jax

# === Setup manipolatore
env = RRPRSingleEnv()
L1, L2, L3, L4, D2 = env.L1_num, env.L2_num, env.L3_num, env.L4_num, env.D2_num
goal_xyz = np.load("results/goal_xyz.npz", allow_pickle=True)['goal']

# === Carica gli stati salvati
states = np.load("results/rrpr_states_over_steps.npz", allow_pickle=True)['states']
Nsteps, Nplot, Hplus1, _ = states.shape

# === Precomputa tutte le traiettorie ee_xyz
all_ee_trajs = []  # shape: (Nsteps, Nplot, H+1, 3)
for step in range(Nsteps):
    step_trajs = []
    for i in range(Nplot):
        ee_seq = []
        for t in range(Hplus1):
            q = states[step][i][t, :4]
            T, *_ = forward_kinematics_rrpr_jax(q, L1, L2, L3, L4, D2)
            pos = np.array(T[:3, 3])
            ee_seq.append(pos)
        step_trajs.append(np.stack(ee_seq))  # (H+1, 3)
    all_ee_trajs.append(np.stack(step_trajs))  # (Nplot, H+1, 3)

# === Setup figura
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1.8, 1.8])
ax.set_ylim([-1.8, 1.8])
ax.set_zlim([-1.5, 0.3])
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.view_init(elev=45, azim=45)
goal_marker = ax.scatter(*goal_xyz, c='red', marker='x', s=60, label='Goal')
ax.legend()

# === Inizializza linee (una per ogni sample)
colors = plt.cm.plasma(np.linspace(0, 1, Nplot))
lines = [ax.plot([], [], [], '-', color=colors[i], alpha=0.8)[0] for i in range(Nplot)]

def update(frame):
    ax.set_title(f"Reverse diffusion step {frame+1}/{Nsteps}")
    step_trajs = all_ee_trajs[frame]  # shape (Nplot, H+1, 3)
    for i in range(Nplot):
        traj = step_trajs[i]
        lines[i].set_data(traj[:, 0], traj[:, 1])
        lines[i].set_3d_properties(traj[:, 2])
    return lines

ani = FuncAnimation(fig, update, frames=Nsteps, interval=300, blit=False)
ani.save("results/rrpr_all_ee_trajs.mp4", fps=3)
print(" Video salvato: results/rrpr_all_ee_trajs.mp4")
# plt.show()  # opzionale
