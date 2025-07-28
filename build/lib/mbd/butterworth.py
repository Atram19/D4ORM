from scipy.signal import freqz
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter
from scipy.signal import butter, lfilter
import os
import jax.numpy as jnp
import jax
path = "results/"
# --- Funzione AR(1) originale ---
def ar1_noise(eta, rho=0.9):
    def step(eps_prev, eta_t):
        return (
            rho * eps_prev + jnp.sqrt(1 - rho**2) * eta_t,
            rho * eps_prev + jnp.sqrt(1 - rho**2) * eta_t,
        )

    def single_sequence(eta_single):  # (H, n, Nu)
        eps0 = eta_single[0]
        _, eps_seq = jax.lax.scan(step, eps0, eta_single[1:])
        return jnp.concatenate([eps0[None], eps_seq], axis=0)

    return jax.vmap(single_sequence)(eta)  # shape (Nsample, H, n, Nu)

def ar1_noise_numpy(key: int, shape, rho=0.9, sigma=1.0):
    """
    key: int seed
    shape: (Nsample, H, n, Nu)
    rho: correlazione temporale
    sigma: deviazione standard del rumore
    """
    np.random.seed(key)
    N, H, n, Nu = shape
    eps = np.zeros((N, H, n, Nu))
    
    # inizializza t=0
    eps[:, 0, :, :] = sigma * np.random.randn(N, n, Nu)

    for t in range(1, H):
        noise_t = sigma * np.random.randn(N, n, Nu)
        eps[:, t, :, :] = rho * eps[:, t - 1, :, :] + np.sqrt(1 - rho**2) * noise_t

    return eps

def get_butterworth_coeffs(order: int, fc: float, fs: float):
    """
    Crea i coefficienti del filtro Butterworth passa-basso normalizzati.
    - order: ordine del filtro
    - fc: frequenza di taglio [Hz]
    - fs: frequenza di campionamento [Hz]
    """
    Wn = fc / (fs / 2)  # normalizza [0,1], dove 1 = Nyquist
    b, a = butter(order, Wn, btype='low')
    return b, a

def butterworth_filter_numpy(eps_u_np, b, a):
    # eps_u_np shape: (Nsample, H, n, Nu)
    Nsample, H, n, Nu = eps_u_np.shape
    eps_u_filt = np.zeros_like(eps_u_np)

    for i in range(Nsample):
        for j in range(n):
            for k in range(Nu):
                eps_u_filt[i, :, j, k] = lfilter(b, a, eps_u_np[i, :, j, k])
    return eps_u_filt


dt = 0.1        # tempo tra due step (tipico in MPPI)
fs = 1 / dt      # = 20 Hz
fc = 0.5         # voglio tagliare sopra i 2 Hz
order = 4

b, a = get_butterworth_coeffs(order, fc, fs)

w, h = freqz(b, a, worN=8000)
f = w * fs / (2 * np.pi)

# plt.plot(f, 20 * np.log10(abs(h)))
# plt.axvline(fc, color='red', linestyle='--', label=f'fc = {fc} Hz')
# plt.title('Risposta in frequenza - filtro Butterworth')
# plt.xlabel('Frequenza [Hz]')
# plt.ylabel('Ampiezza [dB]')
# plt.grid()
# plt.legend()
# plt.savefig(f"{path}/batterworth.png")
# plt.close()


# --- Parametri di base ---
Nsample = 4        # numero di traiettorie
Ndiffuse = 250
H = 10            # orizzonte temporale
n = 1              # numero robot
Nu = 1             # dimensione azione
rho = 0.99          # coeff AR(1)
fc = 0.5           # cutoff frequenza (Hz)
dt = 0.1           # passo temporale
order = 2          # ordine filtro Butterworth
seed = 1
# # --- Generatore rumore AR(1) ---
# def generate_ar1_noise(N, H, rho=0.99):
#     eps = np.zeros((N, H))
#     eps[:, 0] = np.random.randn(N)
#     for t in range(1, H):
#         eps[:, t] = rho * eps[:, t - 1] + np.sqrt(1 - rho**2) * np.random.randn(N)
#     return eps

# # --- Generatore rumore bianco filtrato Butterworth ---
# def butter_lowpass_filter(data, cutoff, fs, order=2):
#     nyq = 0.5 * fs
#     norm_cutoff = cutoff / nyq
#     b, a = butter(order, norm_cutoff, btype='low', analog=False)
#     return lfilter(b, a, data)

# def generate_butter_noise(N, H, fc=2.0, dt=0.1, order=2):
#     raw = np.random.randn(N, H)
#     filtered = np.array([butter_lowpass_filter(traj, fc, fs=1/dt, order=order) for traj in raw])
#     return filtered

# # --- Generazione ---
# np.random.seed(seed)
# noise_white = np.random.randn(Nsample, H)
# noise_ar1 = generate_ar1_noise(Nsample, H, rho=rho)
# noise_butter = butterworth_filter_numpy(eps_u_np, b, a)






# --- Directory output plot ---
path = "results/"
os.makedirs(path, exist_ok=True)




# --- Generazione rumore ---
rng = jax.random.PRNGKey(seed)
rng, rng_eps = jax.random.split(rng)
eta = jax.random.normal(rng_eps, (Nsample, H, n, Nu))  # rumore bianco
eps_u = ar1_noise(eta, rho=rho)  # rumore AR(1)
eta_np = np.array(eta[..., 0, 0])  # shape (Nsample, H)
# --- Conversione in NumPy per il plot ---
# --- Conversione in NumPy per il plot ---
eps_np = np.array(eps_u[..., 0, 0])       # AR(1)
eps_u_np = np.array(eta)                  # rumore bianco da filtrare
noise_butter = butterworth_filter_numpy(eps_u_np, b, a)  # shape (Nsample, H, n, Nu)
noise_butter_np = noise_butter[..., 0, 0]  # shape (Nsample, H)

# --- Plot ---
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

for i in range(Nsample):
    axs[0].plot(eta_np[i], alpha=0.7)
axs[0].set_title("White noise")

for i in range(Nsample):
    axs[1].plot(eps_np[i], alpha=0.7)
axs[1].set_title(f"AR(1) noise (rho={rho})")

for i in range(Nsample):
    axs[2].plot(noise_butter_np[i], alpha=0.7)
axs[2].set_title(f"Low-pass Butterworth noise (fc={fc} Hz)")

plt.xlabel("Time step")
plt.tight_layout()
plt.savefig(f"{path}/rumori.png")
plt.close()


plt.xlabel("Time step")
plt.tight_layout()
plt.savefig(f"{path}/rumori.png")
plt.close()
