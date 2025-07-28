# import jax
# from jax import numpy as jnp
# from flax import struct
# from functools import partial
# import matplotlib.pyplot as plt
# import mbd
# pos = jnp.array([
#   [0.0, 0.0],
#   [1.0, 0.0]
# ])
# print(pos.shape)                # (2, 2)
# print(pos[:, None, :].shape)    # (2, 1, 2)
# print(pos[None, :, :].shape)    # (1, 2, 2)

# diff = pos[:, None, :] - pos[None, :, :]
# print(diff.shape)               # (2, 2, 2)
# print(diff)

import matplotlib.pyplot as plt
import jax
from mbd.envs.multi_car import  Args as args
from scipy.interpolate import BSpline
from scipy.interpolate import make_interp_spline

def make_bspline_basis(H, n_b, degree=5):
    import numpy as np
    knots = np.linspace(0, 1, n_b - degree + 1)**1.5
    knots = np.concatenate(([0] * degree, knots, [1] * degree))  # padded
    t_vals = np.linspace(0, 1, H)
    B = np.stack([BSpline.basis_element(knots[i:i+degree+1])(t_vals) for i in range(n_b)], axis=1)
    return jnp.array(B)  # shape (H, n_b)
def interpolate_spline(W, B):  # W: (nb, n, 2), B: (H, nb)
    return jnp.einsum('tn,bri->tri', B, W)  # returns U: (H, n, 2)
# === Setup base spline e coefficenti casuali ===
nb = 20
B = make_bspline_basis(H=args.Hsample, n_b=nb)
B = B / B.max()  # oppure: B = B / jnp.sum(B, axis=1, keepdims=True)

rng = jax.random.PRNGKey(args.seed + 999)

W_test = jax.random.normal(rng, (nb, args.n_robots, 2)) * 0.1
U_test = interpolate_spline(W_test, B)  # shape: (Hsample, n_robots, 2)

# === Plotta le componenti di controllo per ogni robot ===
fig, axs = plt.subplots(2, args.n_robots, figsize=(4 * args.n_robots, 6))

for i in range(args.n_robots):
    axs[0, i].plot(U_test[:, i, 0])  # velocità lineare
    axs[0, i].set_title(f"Robot {i} - v(t)")
    axs[0, i].grid(True)

    axs[1, i].plot(U_test[:, i, 1])  # velocità angolare
    axs[1, i].set_title(f"Robot {i} - omega(t)")
    axs[1, i].grid(True)

plt.tight_layout()
plt.savefig("results/debug_spline_controls.png")
#plt.show()

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from mbd.envs import MultiCar2d
from mbd.envs.multi_car import Args

args = Args()

def make_bspline_basis(H, n_b, degree=3):
    knots = np.linspace(0, 1, n_b - degree + 1) ** 1.2
    knots = np.concatenate(([0] * degree, knots, [1] * degree))
    t_vals = np.linspace(0, 1, H)
    B = np.stack([BSpline.basis_element(knots[i:i + degree + 1])(t_vals)
                  for i in range(n_b)], axis=1)
    return jnp.array(B / B.max())

def make_spline_towards_goal(x0, xg, B):
    nb = B.shape[1]
    n = x0.shape[0]
    delta = xg[:, :2] - x0[:, :2]
    distance = jnp.linalg.norm(delta, axis=1) + 1e-6
    theta = jnp.arctan2(delta[:, 1], delta[:, 0])
    v = distance / 10.0
    omega = jnp.zeros_like(v)
    U_const = jnp.stack([v, omega], axis=-1)
    U_traj = jnp.tile(U_const[None, :, :], (B.shape[0], 1, 1))
    B_pinv = jnp.linalg.pinv(B)
    W = jnp.einsum("bh,nhi->bni", B_pinv, U_traj[None, ...])
    return W[0]

def interpolate_spline(W, B):
    return jnp.einsum('tn,bri->tri', B, W)

# === Setup test ===
env = MultiCar2d(n=args.n_robots)
B = make_bspline_basis(H=args.Hsample, n_b=20)
W = make_spline_towards_goal(env.x0, env.xg, B)
U = interpolate_spline(W, B)

print("OK fino al rollout")  # se arrivi qui, tutto è corretto

# prova senza rollout
for i in range(args.n_robots):
    plt.plot(U[:, i, 0], label=f"v_{i}")
plt.legend()
plt.savefig("results/test_spline_controls.png")

# === φ(u): base usata da te (Keys 1981)
def phi(u):
    absu = jnp.abs(u)
    return jnp.where(
        absu < 1,
        1.5 * absu**3 - 2.5 * absu**2 + 1,
        jnp.where(
            absu < 2,
            -0.5 * absu**3 + 2.5 * absu**2 - 4 * absu + 2,
            0.0
        )
    )

# === B₃(u): base B-spline classica
def b_spline_cubic(u):
    absu = jnp.abs(u)
    return jnp.where(
        absu < 1,
        (2/3) - absu**2 + 0.5 * absu**3,
        jnp.where(
            (absu >= 1) & (absu < 2),
            (1/6) * (2 - absu)**3,
            0.0
        )
    )

# === Parametri
H = 100
Nknots = 20
t_knots = jnp.linspace(0, H - 1, Nknots)
x_eval = jnp.arange(H)
dx = t_knots[1] - t_knots[0]
u = (x_eval[:, None] - t_knots[None, :]) / dx  # shape (H, Nknots)

# === Matrici di pesi
weights_phi = phi(u)
weights_phi = weights_phi / weights_phi.sum(axis=1, keepdims=True)

weights_bspline = b_spline_cubic(u)
weights_bspline = weights_bspline / weights_bspline.sum(axis=1, keepdims=True)

# === Confronto visivo su una riga specifica
i = 50  # frame centrale
plt.plot(weights_phi[i], label="φ(u) (Keys)")
plt.plot(weights_bspline[i], '--', label="B₃(u) (B-spline classica)")
plt.title(f"Confronto dei pesi di interpolazione – frame {i}")
plt.xlabel("indice nodo k")
plt.ylabel("peso φ_k(t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/confronto_pesi_phi_bspline.png", dpi=150)