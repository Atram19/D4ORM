import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mbd.envs import MultiCar2d
import jax
from mbd.utils import rollout_multi_us
import functools
import os
from mbd.envs.multi_car import  Args
import tyro
from matplotlib.lines import Line2D

# === 1. Cartella dove cercare i file
path = "results/multicar_iterative"

# === Stile uniforme (stesso del render flow) ===
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "DejaVu Serif", "Times"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

candidates = [
    f for f in os.listdir(path)
    if f.startswith("optimized_data_") and f.endswith(".npz")
]

if not candidates:
    raise FileNotFoundError("Nessun file optimized_data_*.npz trovato nella cartella.")

candidates = sorted(
    candidates,
    key=lambda f: os.path.getmtime(os.path.join(path, f))
)

filename = candidates[-1]
print(f"Caricato file più recente: {filename}")

if "LIDEC" in filename:
    ecd_tag = "LIDEC"
elif "d4orm+ecd" in filename:
    ecd_tag = "d4orm+ecd"
elif "LID" in filename:
    ecd_tag = "LID"
if "form" in filename: 
    form = "form"
else:
   form = "_"
data = np.load(os.path.join(path, filename))
traj = data['traj']      
goals = data['goals']    
rewards = data['rewards'] 

n, T, _ = traj.shape

# === 1. Distanza minima tra robot ===
min_distances = [pdist(traj[:, t, :2]).min() for t in range(T)]
plt.figure()
plt.plot(min_distances, lw=1.2, color="#1f77b4")
plt.title("Distanza minima tra i robot nel tempo")
plt.xlabel("Tempo [step]")
plt.ylabel("Distanza minima [m]")
plt.grid(True, linestyle="-", alpha=0.6)
plt.tight_layout()
plt.savefig(f"{path}/plot_min_distance_{ecd_tag}_{form}.png")
plt.close()

# 2. Errore finale rispetto al goal
# === 2. Errore finale rispetto al goal ===
final_positions = traj[:, -1, :2]
goal_positions = goals[:, :2]
errors = np.linalg.norm(final_positions - goal_positions, axis=1)
plt.figure()
plt.bar(range(n), errors, color="#2ca02c")
plt.title("Errore finale rispetto al goal")
plt.xlabel("Robot")
plt.ylabel("Errore [m]")
plt.grid(True, linestyle="-", alpha=0.6)
plt.tight_layout()
plt.savefig(f"{path}/plot_goal_errors_{ecd_tag}_{form}.png")
plt.close()


# === 3. Reward medio per iterazione ===
K = rewards.shape[0]
plt.figure()
for i in range(n):
    plt.plot(range(K), rewards[:, i], label=f"Robot {i}")
plt.title("Reward medio per robot nelle iterazioni")
plt.xlabel("Iterazione")
plt.ylabel("Reward")
plt.grid(True, linestyle="-", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(f"{path}/plot_rewards_{ecd_tag}_{form}.png")
plt.close()

# === 4. Distanza dal goal nel tempo ===
errors_over_time = np.zeros((n, T))
for i in range(n):
    for t in range(T):
        errors_over_time[i, t] = np.linalg.norm(traj[i, t, :2] - goal_positions[i])
plt.figure()
for i in range(n):
    plt.plot(errors_over_time[i], label=f"Robot {i}")
plt.title("Distanza dal goal nel tempo")
plt.xlabel("Tempo [step]")
plt.ylabel("Errore [m]")
plt.grid(True, linestyle="-", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(f"{path}/plot_goal_error_over_time_{ecd_tag}_{form}.png")
plt.close()



# # === VIDEO GLOBAL DIFFUSION ===
# # === Parametri e caricamento dati
# path = "results/multicar_iterative/risutati"
# global_file = os.path.join(path, "global_diffusion_data.npz")

# if os.path.exists(global_file):
#     print("Generazione video reverse diffusion globale...")
#     data = np.load(global_file)

#     sample_trajectories_xy = data["sample_trajectories_xy"]  # (T, Nsample, H, n, 2)
#     Ybar_list = data["Ybar_list"] if "Ybar_list" in data else None




# # === Configurazione ===
# output_path = "results/multicar_iterative/global_diffusion_video.mp4"
# os.makedirs(os.path.dirname(output_path), exist_ok=True)

# data = np.load("results/multicar_iterative/global_Yi_list.npz")

# trajectories_all = data["trajectories_samples"]
# trajectories_denoised = data["trajectories_denoised"]
# n =Args.n_robots
# cmap = plt.get_cmap("tab20", n)
# fig, ax = plt.subplots(figsize=(6, 6))
# args = tyro.cli(Args)
# env = MultiCar2d(n=args.n_robots, formation_shift=args.formation_shift,ECD=args.ECD, obstacles_enabled=args.obstacles_enabled)

# def update(frame):
#     ax.clear()
#     ax.set_title(f"Reverse Diffusion Step {frame}")
#     ax.set_xlim(-5, 5)
#     ax.set_ylim(-5, 5)
#     ax.set_aspect("equal")
#     # for x_c, y_c, w, h in env.static_obstacles:
#     #         rect = plt.Rectangle((x_c - w / 2, y_c - h / 2), w, h,
#     #                             linewidth=1, edgecolor='red', facecolor='red', alpha=0.5)
#     #         ax.add_patch(rect)
#     buffer_min = 0.2
#     buffer_max = 0.5

#     for x_c, y_c, w, h in env.static_obstacles:
#         rect_outer = plt.Rectangle(
#             (x_c - (w / 2 + buffer_max), y_c - (h / 2 + buffer_max)),
#             w + 2 * buffer_max,
#             h + 2 * buffer_max,
#             linewidth=0,
#             facecolor='yellow',
#             alpha=0.1,
#             zorder=1
#         )
#         ax.add_patch(rect_outer)

#     for x_c, y_c, w, h in env.static_obstacles:
#         rect_inner = plt.Rectangle(
#             (x_c - (w / 2 + buffer_min), y_c - (h / 2 + buffer_min)),
#             w + 2 * buffer_min,
#             h + 2 * buffer_min,
#             linewidth=0,
#             facecolor='yellow',
#             alpha=0.5,
#             zorder=2
#         )
#         ax.add_patch(rect_inner)

#     for x_c, y_c, w, h in env.static_obstacles:
#         rect_real = plt.Rectangle(
#             (x_c - w / 2, y_c - h / 2),
#             w, h,
#             linewidth=1,
#             edgecolor='red',
#             facecolor='red',
#             zorder=3
#         )
#         ax.add_patch(rect_real)


#     # Campioni (trasparenti)
#     samples = trajectories_all[frame]  # shape (Nsample, T+1, n, 2)
#     #print("samples[s].shape:", samples[s].shape)

#     for i in range(n):
#         for s in range(min(100, samples.shape[0])):  # Limita a 100 campioni per leggibilità
#             traj = samples[s]  # shape: (T+1, n, 2)
#             x = traj[:, i, 0]
#             y = traj[:, i, 1]
#             ax.plot(x, y, alpha=0.1, color=cmap(i))

#     # Traiettoria ottimizzata
#     traj_opt = trajectories_denoised[frame]  # shape: (T+1, n, 2)
   

#     for i in range(n):
#         x_opt = traj_opt[:, i, 0]
#         y_opt = traj_opt[:, i, 1]
#         ax.plot(x_opt, y_opt, color=cmap(i), linewidth=2.0)

#         # Optional: marker inizio/fine
#         ax.plot(x_opt[0], y_opt[0], "o", color=cmap(i), markersize=4)  # start
#         ax.plot(x_opt[-1], y_opt[-1], "s", color=cmap(i), markersize=4)  # end

#     return []

# # === Crea animazione e salva ===
# ani = animation.FuncAnimation(fig, update, frames=len(trajectories_all), interval=150)
# ani.save(output_path, fps=5, dpi=150)
# print(f" Video salvato: {output_path}")

# === VIDEO REVERSE DIFFUSION GLOBALE ===
output_path = os.path.join(path, "global_diffusion_video.mp4")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

data = np.load(os.path.join(path, "global_Yi_list.npz"))
trajectories_all = data["trajectories_samples"]
trajectories_denoised = data["trajectories_denoised"]

args = tyro.cli(Args)
env = MultiCar2d(
    n=args.n_robots,
    formation_shift=args.formation_shift,
    ECD=args.ECD,
    obstacles_enabled=args.obstacles_enabled,
)

# === Funzione per disegnare le zone di penalità ===
def draw_obstacle_penalty_zones(ax, env):
    buffer_min, buffer_max = 0.2, 0.5
    for x_c, y_c, w, h in env.static_obstacles:
        # Zona esterna
        rect = plt.Rectangle(
                (x_c - w / 2, y_c - h / 2), w, h,
                linewidth=1.0, edgecolor='black',
                facecolor='#d3d3d3', zorder=1
            )
        ax.add_patch(rect)
        # Zona interna
        rect_outer = plt.Rectangle(
                (x_c - (w / 2 + buffer_max), y_c - (h / 2 + buffer_max)),
                w + 2 * buffer_max, h + 2 * buffer_max,
                linewidth=0.8, edgecolor='none',
                facecolor='#a6bddb', alpha=0.25, zorder=1
            )
        ax.add_patch(rect_outer)

        # Ostacolo reale ben visibile
        rect_inner = plt.Rectangle(
                (x_c - (w / 2 + buffer_min), y_c - (h / 2 + buffer_min)),
                w + 2 * buffer_min, h + 2 * buffer_min,
                linewidth=0.8, edgecolor='none',
                facecolor='#3690c0', alpha=0.35, zorder=2
            )
        ax.add_patch(rect_inner)
        

# === Animazione ===
n = Args.n_robots
cmap = plt.get_cmap("tab20", n)
fig, ax = plt.subplots(figsize=(6, 6))

def update(frame):
    ax.clear()
    ax.set_title(f"Reverse Diffusion Step {frame}")
    palette = [
        "#1f77b4",  # blu
        "#ff7f0e",  # arancio
        "#2ca02c",  # verde
        "#d62728",  # rosso
        "#9467bd",  # viola
        "#8c564b",  # marrone
        "#e377c2",  # rosa chiaro
        "#7f7f7f",  # grigio
    ]
    

    ax.set_aspect("equal")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.grid(
                True, linestyle="-", color="k", linewidth=0.6, alpha=0.7
            )  
    draw_obstacle_penalty_zones(ax, env)

    samples = trajectories_all[frame]
    for i in range(n):
        for s in range(min(80, samples.shape[0])):
            traj = samples[s]
            ax.plot(traj[:, i, 0], traj[:, i, 1], alpha=0.07, color=palette[i])

    traj_opt = trajectories_denoised[frame]
    for i in range(n):
        ax.plot(traj_opt[:, i, 0], traj_opt[:, i, 1], color=palette[i], linewidth=2)
        ax.plot(traj_opt[0, i, 0], traj_opt[0, i, 1], "o", color=palette[i], markersize=4)
       # === Legenda compatta ===
    

    return []

ani = animation.FuncAnimation(fig, update, frames=len(trajectories_all), interval=150)
# === SALVATAGGIO IMMAGINI STATICHE (iniziale e finale) ===
def save_static_diffusion_images(trajectories_all, trajectories_denoised, env, path, cmap):
    """
    Salva due figure statiche: step iniziale e step finale della reverse diffusion globale.
    """
    steps = {"iniziale": 0, "finale": len(trajectories_all) - 1}
     # === Palette accademica e sobria ===
    palette = [
        "#1f77b4",  # blu
        "#ff7f0e",  # arancio
        "#2ca02c",  # verde
        "#d62728",  # rosso
        "#9467bd",  # viola
        "#8c564b",  # marrone
        "#e377c2",  # rosa chiaro
        "#7f7f7f",  # grigio
    ]
    

    for tag, idx in steps.items():
        fig, ax = plt.subplots(figsize=(6, 6))
        
        ax.set_aspect("equal")
        ax.grid(
                True, linestyle="-", color="k", linewidth=0.6, alpha=0.7
            )  
        draw_obstacle_penalty_zones(ax, env)

        n = env.n
        samples = trajectories_all[idx]
        for i in range(n):
            for s in range(min(80, samples.shape[0])):
                traj = samples[s]
                ax.plot(traj[:, i, 0], traj[:, i, 1], alpha=0.07, color=palette[i])

        traj_opt = trajectories_denoised[idx]
          # Posizione goal (stella o quadrato vuoto)
       
        for i in range(n):
            gx, gy = env.xg[i, 0], env.xg[i, 1]
            ax.plot(
                gx, gy, marker='s', color=palette[i], markersize=5.5,
                    markeredgewidth=0.8, zorder=5
            )
            ax.plot(traj_opt[:, i, 0], traj_opt[:, i, 1], color=palette[i], linewidth=2)
            ax.plot(traj_opt[0, i, 0], traj_opt[0, i, 1], "o", color=palette[i], markersize=4)

        ax.set_title(f"Reverse Diffusion – Step {tag}", pad=6)
        legend_elements = [
        Line2D([0], [0], color='gray', lw=1, alpha=0.3, label='Campioni stocastici'),
        Line2D([0], [0], color='k', lw=2, label='Traiettoria ottimizzata'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k',
                markersize=5, label='Posizione iniziale'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='k',
                markersize=5, label='Posizione finale'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(path, f"global_diffusion_{tag}.png"), dpi=300)
        plt.close(fig)

    print("Immagini statiche salvate: global_diffusion_iniziale.png, global_diffusion_finale.png")


# === Genera immagini prima del video ===
save_static_diffusion_images(trajectories_all, trajectories_denoised, env, path, cmap)

ani.save(output_path, fps=5, dpi=150)
print(f" Video salvato: {output_path}")