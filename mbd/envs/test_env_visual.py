import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from multi_car import MultiCar2d, car_dynamics
import matplotlib
matplotlib.use("Agg")  # Usa un backend che non richiede GUI
import matplotlib.pyplot as plt

# 1. Parametri di test
n_robots = 4
H=100
# 2. Inizializza ambiente
env = MultiCar2d(n=n_robots)
S = env.x0  # stato iniziale (n, 3)
print(f"Stato iniziale: {S}")
# 3. Campiona controlli casuali: U ~ N(0, 0.1)
key = jax.random.PRNGKey(0)
U = 0.2 * jax.random.normal(key, shape=(n_robots, H, env.action_size))

# 4. Rollout: integra i controlli per ogni robot
def rollout(S, U, dt):
    def step(x, u_seq):
        def rk_step(x_t, u_t):
            return rk4(car_dynamics, x_t, u_t, dt), None
        X, _ = jax.lax.scan(rk_step, x, u_seq)
        return jnp.vstack([x[None], X])
    return jax.vmap(step)(S, U)  # → (n, H+1, 3)

# (4b) definisci dinamica se non accessibile direttamente
def rk4(dynamics, x, u, dt):
    k1 = dynamics(x, u)
    k2 = dynamics(x + dt / 2 * k1, u)
    k3 = dynamics(x + dt / 2 * k2, u)
    k4 = dynamics(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

# 5. Calcola traiettorie
X = rollout(S, U, env.dt)  # (n, H+1, 3)

# 6. Visualizza
# # fig, ax = plt.subplots(figsize=(8, 8))
# # env.render(ax, X, goals=env.xg)
# # plt.title("Traiettorie dei robot - MultiCar2d")
# # plt.show()
fig, ax = plt.subplots(figsize=(8, 8))
env.render(ax, X, goals=env.xg)
plt.title("Traiettorie dei robot - MultiCar2d")
plt.savefig("output_traj.png")
print(" Grafico salvato in output_traj.png")
