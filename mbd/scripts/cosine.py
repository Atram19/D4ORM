import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
def cosine_beta_schedule(T, s=0.008):
    """
    Cosine schedule per betas, come in Improved DDPM
    Restituisce un array di T valori beta_t ∈ (0,1)
    """
    t = jnp.arange(T + 1, dtype=jnp.float32)
    f_t = jnp.cos(((t / T + s) / (1 + s)) * jnp.pi / 2) ** 2
    alphas_bar = f_t / f_t[0]
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    betas = 1 - alphas
    return jnp.clip(betas, a_min=1e-5, a_max=0.999)

def cosine_beta_schedule_scaled(T, beta0, betaT, s=0.008):
    """
    Cosine schedule con scaling per rendere beta0 e betaT coerenti con valori desiderati.
    """
    t = jnp.arange(T + 1, dtype=jnp.float32)
    f_t = jnp.cos(((t / T + s) / (1 + s)) * jnp.pi / 2) ** 2
    alphas_bar = f_t / f_t[0]
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    betas = 1 - alphas

    # Riscalamento lineare: beta0_target → beta0, betaT_target → betaT
    beta_min, beta_max = betas.min(), betas.max()
    betas_scaled = (betas - beta_min) / (beta_max - beta_min)  # ∈ [0,1]
    betas_scaled = betas_scaled * (betaT - beta0) + beta0      # ∈ [beta0, betaT]

    return betas_scaled
# Parameters
T = 100
beta0 = 1e-4
betaT = 1e-2

# Generate schedules
betas_lin = jnp.linspace(beta0, betaT, T)
betas_cos = cosine_beta_schedule(T)
betas_cos_scaled = cosine_beta_schedule_scaled(T, beta0, betaT)

# Compute cumulative alphas_bar and sigmas
def compute_alpha_sigma(betas):
    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)
    sigmas = jnp.sqrt(1 - alphas_bar)
    snr = alphas_bar / (1 - alphas_bar)
    return alphas_bar, sigmas, snr

ab_lin, sig_lin, snr_lin = compute_alpha_sigma(betas_lin)
ab_cos, sig_cos, snr_cos = compute_alpha_sigma(betas_cos)
ab_cos_scaled, sig_cos_scaled, snr_cos_scaled = compute_alpha_sigma(betas_cos_scaled)

# Trasforma jnp in np per il plotting
ab_lin_np = np.array(ab_lin)
ab_cos_np = np.array(ab_cos)

# Normalizza l'asse x in [0, 1]
timesteps = np.linspace(0, 1, len(ab_lin_np))

plt.figure(figsize=(6, 4))
plt.plot(timesteps, ab_lin_np, label="linear", color='tab:blue')
plt.plot(timesteps, ab_cos_np, label="cosine", color='darkorange')

plt.xlabel("diffusion step (t/T)")
plt.ylabel(r"$\bar{\alpha}_t$")
plt.title("Comparison of Linear and Cosine Noise Schedules")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figure5_cosine_schedule.png", dpi=300)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

def cosine_beta_schedule(T, s=0.008):
    t = np.arange(T + 1, dtype=np.float32)
    f_t = np.cos(((t / T + s) / (1 + s)) * np.pi / 2) ** 2
    alphas_bar = f_t / f_t[0]
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    betas = 1 - alphas
    return np.clip(betas, a_min=1e-5, a_max=0.999)

def cosine_beta_schedule_scaled(T, beta0, betaT, s=0.008):
    betas = cosine_beta_schedule(T, s)
    beta_min, beta_max = betas.min(), betas.max()
    betas_scaled = (betas - beta_min) / (beta_max - beta_min)
    betas_scaled = betas_scaled * (betaT - beta0) + beta0
    return betas_scaled

# Parameters
T = 100
beta0 = 1e-4
betaT = 1e-2

# Generate schedules
betas_lin = np.linspace(beta0, betaT, T)
betas_cos = cosine_beta_schedule(T)
betas_cos_scaled = cosine_beta_schedule_scaled(T, beta0, betaT)

# Compute cumulative alphas_bar and sigmas
def compute_alpha_sigma(betas):
    alphas = 1.0 - betas
    alphas_bar = np.cumprod(alphas)
    sigmas = np.sqrt(1 - alphas_bar)
    snr = alphas_bar / (1 - alphas_bar)
    return alphas_bar, sigmas, snr

ab_lin, sig_lin, snr_lin = compute_alpha_sigma(betas_lin)
ab_cos, sig_cos, snr_cos = compute_alpha_sigma(betas_cos)
ab_cos_scaled, sig_cos_scaled, snr_cos_scaled = compute_alpha_sigma(betas_cos_scaled)

# Plotting Betas
fig_beta, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
axs[0].plot(betas_lin, color="blue")
axs[0].set_title("Beta Schedule - Linear")
axs[0].set_ylabel("beta_t")
axs[0].grid()

axs[1].plot(betas_cos, color="green")
axs[1].set_title("Beta Schedule - Cosine")
axs[1].set_ylabel("beta_t")
axs[1].grid()

axs[2].plot(betas_cos_scaled, color="red")
axs[2].set_title("Beta Schedule - Cosine Scaled")
axs[2].set_ylabel("beta_t")
axs[2].set_xlabel("Timestep")
axs[2].grid()

fig_beta.suptitle("Beta Schedules Comparison", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("beta_cosine.png", dpi=300)

# Plotting Sigmas
fig_sigma, ax_sigma = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
ax_sigma[0].plot(sig_lin, color="blue")
ax_sigma[0].set_title("Sigma_t - Linear")
ax_sigma[0].set_ylabel("sigma_t")
ax_sigma[0].grid()

ax_sigma[1].plot(sig_cos, color="green")
ax_sigma[1].set_title("Sigma_t - Cosine")
ax_sigma[1].set_ylabel("sigma_t")
ax_sigma[1].grid()

ax_sigma[2].plot(sig_cos_scaled, color="red")
ax_sigma[2].set_title("Sigma_t - Cosine Scaled")
ax_sigma[2].set_ylabel("sigma_t")
ax_sigma[2].set_xlabel("Timestep")
ax_sigma[2].grid()

fig_sigma.suptitle("Sigma_t (sqrt(1 - alpha_bar)) Comparison", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("sigma_cosine.png", dpi=300)
