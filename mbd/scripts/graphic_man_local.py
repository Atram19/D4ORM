import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mbd.envs.class_manipulator import RRPRSingleEnv, forward_kinematics_rrpr_jax, rollout_single_us
import jax
from functools import partial

# === Setup env e FK
env = RRPRSingleEnv()
L1, L2, L3, L4, D2 = env.L1_num, env.L2_num, env.L3_num, env.L4_num, env.D2_num

# === Goal
goal_xyz = np.load("results/goal_xyz.npz", allow_pickle=True)['goal']

# === Carica traiettorie U ottime
U_ks = np.load("results/rrpr_U_local.npz", allow_pickle=True)['U_ks']  # (K+1, H, Nu)
K_plus_1, H, Nu = U_ks.shape
K = K_plus_1 - 1

# === Carica campioni locali [k][w] = (Nplot, H+1, D)
states_local = np.load("results/rrpr_states_local.npz", allow_pickle=True)['states_local']
assert len(states_local) == K, "Numero iterazioni non combacia"

W = len(states_local[0])  # numero finestre per iterazione (assunto costante)
Nplot, Hplus1, D = states_local[0][0].shape
print("Salvo", len(states_local), "iterazioni con", len(states_local[0]), "finestre ciascuna")
print("Salvo", len(U_ks), "traiettorie ottime (U_ks)")

# === Rollout U -> EE
reset_env = jax.jit(env.reset)
step_env = jax.jit(env.step)
state_init = reset_env(jax.random.PRNGKey(0))
rollout_fn = jax.jit(partial(rollout_single_us, step_env, state_init))

def to_ee_xyz(U):
    rew, pipeline, _ = rollout_fn(U)
    q_seq = np.array(pipeline)[:, :4]
    return np.array([np.array(forward_kinematics_rrpr_jax(q, L1, L2, L3, L4, D2)[0][:3, 3]) for q in q_seq])

def batch_to_ee_xyz(batch_q):  # (Nplot, H+1, D)
    return [np.array([np.array(forward_kinematics_rrpr_jax(q, L1, L2, L3, L4, D2)[0][:3, 3]) for q in traj]) for traj in batch_q]

ee_full = [to_ee_xyz(U_ks[k]) for k in range(K)]
print(f"U_ks shape: {U_ks.shape}")  # (11, H, Nu)
print(f"Sto creando ee_full con K = {K}")

# === Setup figura
# === Setup figura
fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.view_init(elev=35, azim=35)
ax.grid(True)

# Limiti (un po' più larghi per non tagliare le traiettorie)
ax.set_xlim([-1.8, 1.8])
ax.set_ylim([-2.2, 1.2])
ax.set_zlim([-1.6, 0.2])
# proporzioni più equilibrate
ax.set_box_aspect((1.8-(-1.8), 1.2-(-2.2), 0.2-(-1.6)))

ax.view_init(elev=35, azim=30)  # buona inclinazione per RRPR

# Goal ben visibile
ax.scatter(*goal_xyz, c='red', marker='o', linewidths=1, label='Goal')
ax.legend(loc='upper right')

# === Linea ottima (molto visibile)
line_opt, = ax.plot([], [], [], '-', color='black', linewidth=3.2, label='Traiettoria ottima', zorder=5)


# === Campioni: colori vividi e marker inizio/fine
cmap = plt.colormaps.get_cmap("tab20")   # "hsv" o "nipy_spectral" per ancora più varietà
colors = [cmap(i % cmap.N) for i in range(Nplot)]

sample_lines   = [ax.plot([], [], [], '-', color=colors[i], alpha=0.95, linewidth=1.8, zorder=4)[0] for i in range(Nplot)]
sample_starts  = [ax.plot([], [], [], 'o', color=colors[i], alpha=0.95, markersize=5, zorder=6)[0] for i in range(Nplot)]
sample_ends    = [ax.plot([], [], [], 'x', color=colors[i], alpha=0.95, markersize=5, zorder=6)[0] for i in range(Nplot)]

title = ax.set_title("")

def init():
    line_opt.set_data([], []); line_opt.set_3d_properties([])

    for ln in sample_lines + sample_starts + sample_ends:
        ln.set_data([], []); ln.set_3d_properties([])
    return [line_opt, *sample_lines, *sample_starts, *sample_ends, ]

def update(frame):
    k = frame // W
    w = frame % W

    # Traiettoria ottima aggiornata alla k-esima iterazione (usa ee_full[k], lungo H)
    traj_opt = ee_full[k]                       # shape (H, 3)
    x, y, z = traj_opt[:, 0], traj_opt[:, 1], traj_opt[:, 2]
    line_opt.set_data(x, y); line_opt.set_3d_properties(z)

   
    # Campioni della finestra w
    samples = batch_to_ee_xyz(states_local[k][w])  # lista di Nplot array (H+1, 3)
    n_show = min(Nplot, len(samples))

    # aggiorna i primi n_show campioni
    for i in range(n_show):
        Xi = samples[i][:, 0]; Yi = samples[i][:, 1]; Zi = samples[i][:, 2]
        sample_lines[i].set_data(Xi, Yi); sample_lines[i].set_3d_properties(Zi)

        # marker start/end
        sample_starts[i].set_data([Xi[0]], [Yi[0]])
        sample_starts[i].set_3d_properties([Zi[0]])

        sample_ends[i].set_data([Xi[-1]], [Yi[-1]])
        sample_ends[i].set_3d_properties([Zi[-1]])


       

        # massimizza contrasto del campione corrente
        sample_lines[i].set_alpha(0.98); sample_lines[i].set_linewidth(2.2)

    # disattiva eventuali linee in eccesso (se n_show < Nplot)
    for i in range(n_show, Nplot):
        for obj in (sample_lines[i], sample_starts[i], sample_ends[i]):
            obj.set_data([], []); obj.set_3d_properties([])

    title.set_text(f"Iterazione {k} • Finestra {w+1}/{W}")
    return [line_opt, *sample_lines, *sample_starts, *sample_ends]

TOTAL_FRAMES = K * W
ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES, init_func=init, interval=400, blit=False)
ani.save("results/rrpr_local_all_windows.mp4", fps=3)
print("✅ Video salvato: results/rrpr_local_all_windows.mp4")