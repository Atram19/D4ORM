import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
import matplotlib.animation as animation

# === 1. Cartella dove cercare i file
path = "results/multicar_iterative"

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

if "ecd" in filename:
    ecd_tag = "ecd"
elif "d4orm+ecd" in filename:
    ecd_tag = "d4orm+ecd"
elif "d4orm" in filename:
    ecd_tag = "d4orm"
if "form" in filename: 
    form = "form"
else:
   form = "_"
data = np.load(os.path.join(path, filename))
traj = data['traj']      
goals = data['goals']    
rewards = data['rewards'] 

n, T, _ = traj.shape

# 1. Distanza minima tra robot nel tempo
min_distances = []
for t in range(T):
    pos = traj[:, t, :2]
    dists = pdist(pos)
    min_distances.append(dists.min())

plt.figure()
plt.plot(min_distances)
plt.title("Distanza minima tra i robot nel tempo")
plt.xlabel("Tempo [step]")
plt.ylabel("Distanza minima [m]")
plt.grid(True)
plt.savefig(f"{path}/plot_min_distance_{ecd_tag}_{form}.png")
plt.close()

# 2. Errore finale rispetto al goal
final_positions = traj[:, -1, :2]
goal_positions = goals[:, :2]
errors = np.linalg.norm(final_positions - goal_positions, axis=1)

plt.figure()
plt.bar(range(n), errors)
plt.title("Errore finale rispetto al goal")
plt.xlabel("Robot")
plt.ylabel("Errore [m]")
plt.grid(True)
plt.savefig(f"{path}/plot_goal_errors_{ecd_tag}_{form}.png")
plt.close()

# 3. Reward per robot durante le iterazioni
K = rewards.shape[0]
plt.figure()
for i in range(n):
    plt.plot(range(K), rewards[:, i], label=f"Robot {i}")
plt.title("Reward medio per robot nelle iterazioni")
plt.xlabel("Iterazione")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()
plt.savefig(f"{path}/plot_rewards_{ecd_tag}_{form}.png")
plt.close()

# 4. Errore rispetto al goal nel tempo
errors_over_time = np.zeros((n, T))
for i in range(n):
    for t in range(T):
        pos = traj[i, t, :2]
        goal = goal_positions[i]
        errors_over_time[i, t] = np.linalg.norm(pos - goal)

plt.figure()
for i in range(n):
    plt.plot(errors_over_time[i], label=f"Robot {i}")
plt.title("Distanza dal goal nel tempo")
plt.xlabel("Tempo [step]")
plt.ylabel("Errore [m]")
plt.grid(True)
plt.legend()
plt.savefig(f"{path}/plot_goal_error_over_time_{ecd_tag}_{form}.png")
plt.close()


path = "results/multicar_iterative"
filename = os.path.join(path, "trend_samples_iter_7.npz")

if os.path.exists(filename):
    data = np.load(filename)
    R_window = data["R_window"]
    J_goal = data["J_goal"]
    J_barrier = data["J_barrier"]
    J_control = data["J_control"]
    H_norm = data["H_norm"]
    noise_norm = data["noise_norm"] 
    output_dir = os.path.join(path, "plot_costandrews")
    os.makedirs(output_dir, exist_ok=True)
    def plot_costandrews(mat, title, ylabel, filename):
        T, N = mat.shape
        x = np.arange(T)
        plt.figure(figsize=(8, 4))

        # Linee sottili per ogni sample
        for i in range(N):
            plt.plot(x, mat[:, i], alpha=0.1, color='blue')

        # Media evidenziata
        plt.plot(x, mat.mean(axis=1), lw=2, color='black', label="Media")

        plt.title(title)
        plt.xlabel("Finestra temporale")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    plot_costandrews(R_window, "Reward nella finestra", "Reward", "R_window.png")
    plot_costandrews(J_goal, "Costo tracking goal", "Costo", "J_goal.png")
    plot_costandrews(J_barrier, "Costo barriera", "Costo", "J_barrier.png")
    plot_costandrews(J_control, "Costo di controllo", "Costo", "J_control.png")

    # plot_costandrews(J_control, "Costo controllo", "Costo", "J_control.png")
    plot_costandrews(H_norm, "Norma dei vincoli", "||h||", "H_norm.png")
   
    
    plt.figure()
    plt.plot(noise_norm)
    plt.title("Norma media del rumore introdotto per finestra")
    plt.xlabel("Finestra")
    plt.ylabel("Norma media del rumore")
    plt.grid(True)
    plt.savefig(f"{path}/plot_noise_norm_{ecd_tag}_{form}.png")
    plt.close()


else:
    print(f" File non trovato: {filename}")



# === VIDEO GLOBAL DIFFUSION ===
# === Parametri e caricamento dati
path = "results/multicar_iterative"
global_file = os.path.join(path, "global_diffusion_data.npz")

if os.path.exists(global_file):
    print("Generazione video reverse diffusion globale...")
    data = np.load(global_file)

    sample_trajectories_xy = data["sample_trajectories_xy"]  # (T, Nsample, H, n, 2)
    Ybar_list = data["Ybar_list"] if "Ybar_list" in data else None

    T, Nsample, H, n, _ = sample_trajectories_xy.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.get_cmap("tab10", n)

    def animate_global(t):
        ax.clear()
        ax.set_title(f"Reverse Diffusion Step {t}")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
        ax.grid(True)

        # Disegna solo 30 sample per evitare confusione
        for traj in sample_trajectories_xy[t][:30]:  # (H, n, 2)
            for r in range(n):
                xy = traj[:, r]
                ax.plot(xy[:, 0], xy[:, 1], alpha=0.1, color=cmap(r))

        # Disegna la traiettoria media Ybar
        # if Ybar_list is not None:
        #     Ybar = Ybar_list[t]  # (H, n, 2)
        #     for r in range(n):
        #         xy = Ybar[:, r]
        #         ax.plot(xy[:, 0], xy[:, 1], color=cmap(r), linewidth=2.0)

    anim = animation.FuncAnimation(fig, animate_global, frames=range(len(sample_trajectories_xy) - 1, -1, -1), interval=400)
    anim.save(os.path.join(path, "global_diffusion_video.mp4"), fps=2, dpi=150)
    plt.close()
    print(" Video salvato: global_diffusion_video.mp4")
    if "reward_terms" in data:
        reward_terms = data["reward_terms"]  # (T, Nsample, n, 6)
        T, Nsample, n, n_terms = reward_terms.shape
        terms_labels = ["r_goal", "r_safe", "r_form", "r_control", "r_obs", "r_total"]
        colors = ["blue", "red", "green", "orange", "purple", "black"]

        # Calcola media su sample e robot → (T, 6)
        reward_mean = reward_terms.mean(axis=(1, 2))

        # Plot di ciascun termine nel tempo
        plt.figure(figsize=(10, 6))
        for i in range(n_terms):
            plt.plot(range(T), reward_mean[:, i], label=terms_labels[i], color=colors[i])
        plt.title("Reward Terms - Reverse Diffusion")
        plt.xlabel("Reverse step")
        plt.ylabel("Valore medio")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{path}/plot_reward_terms_global_{ecd_tag}_{form}.png")
        plt.close()
    else:
        print("reward_terms non presente nel file global_diffusion_data.npz.")
        data = np.load(global_file)
    if "reward_traj_opt" in data:
        reward_traj_opt = data["reward_traj_opt"]  # shape (H, n, 6)
        H, n, _ = reward_traj_opt.shape
        labels = ["r_goal", "r_safe", "r_form", "r_control", "r_obstacles", "r_total"]

        for i, label in enumerate(labels):
            plt.figure()
            for j in range(n):
                plt.plot(reward_traj_opt[:, j, i], label=f"Robot {j}", alpha=0.5)
            plt.plot(reward_traj_opt[:, :, i].mean(axis=1), label="Media", color="black", linewidth=2)
            plt.title(f"{label} nel tempo (traiettoria ottimale)")
            plt.xlabel("Frame")
            plt.ylabel("Valore")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{path}/plot_{label}_over_time_{ecd_tag}_{form}.png")
            plt.close()


else:
    print("File global_diffusion_data.npz non trovato: skip video globale.")

# === VIDEO LOCAL DIFFUSION ===

path = "results/multicar_iterative"
local_file = os.path.join(path, "trend_samples_iter_7.npz")

if os.path.exists(local_file):
    print("Generazione video ottimizzazione locale...")
    data = np.load(local_file, allow_pickle=True)

    trajectories_xy = list(data["trajectories_xy"])           # (N_frames, Nsample, L, n, 2)
    trajectory_buffer = list(data["trajectory_buffer"])        # (N_frames, n, H, 2)
    gradients_buffer = list(data["gradients_buffer"])
    N_frames = len(trajectories_xy)
    Nsample, L, n, _ = trajectories_xy[0].shape
    _, H, _ = trajectory_buffer[0].shape

    fig, ax = plt.subplots(figsize=(5, 5))
    cmap = plt.get_cmap("tab10", n)

    def animate_local(f_idx):
        ax.clear()
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_title(f"Local Diffusion Step {f_idx}")

        # --- 1. Traiettorie campionate ---
        trajs = trajectories_xy[f_idx]  # (Nsample, L, n, 2)
        for k in range(min(80, Nsample)):
            for i in range(n):
                xy = trajs[k, :, i]  # (L, 2)
                ax.plot(xy[:, 0], xy[:, 1], color=cmap(i), alpha=0.2, linewidth=0.7)

        # --- 2. Traiettoria ottimizzata (in evidenza) ---
        traj_opt = trajectory_buffer[f_idx]  # (n, H, 2)
        for i in range(n):
            xy = traj_opt[i]
            ax.plot(xy[:, 0], xy[:, 1], '-', color=cmap(i), linewidth=2)
            ax.plot(xy[0, 0], xy[0, 1], 's', color=cmap(i), markersize=4)
            ax.plot(xy[-1, 0], xy[-1, 1], '*', color=cmap(i), markersize=7)
      
        



    # === ANIMAZIONE ===
    ani = animation.FuncAnimation(fig, animate_local, frames=N_frames, interval=400)
    output_path = os.path.join(path, "local_diffusion_video.mp4")
    ani.save(output_path, fps=2, dpi=150)
    plt.close()
    print(f" Video salvato: {output_path}")