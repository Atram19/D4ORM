import numpy as np
import matplotlib.pyplot as plt
from roboticstoolbox import DHRobot, RevoluteDH, PrismaticDH
from spatialmath import SE3
import os 

# === Funzioni utili ===


def I_bar_along_z(m, L):
    return np.diag([0, m * L**2 / 12,m * L**2 / 12])

def rk4(dynamics, x, u, dt):
    k1 = dynamics(x, u)
    k2 = dynamics(x + dt/2 * k1, u)
    k3 = dynamics(x + dt/2 * k2, u)
    k4 = dynamics(x + dt * k3, u)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def robot_dynamics(x, u):
    q = x[:4]
    qd = x[4:]
    # robot.links[2].r = [0, 0, q[2]/2 + 2]  # aggiorna CoM prismatico

    B = robot.inertia(q)
    C = robot.coriolis(q, qd)
    G = robot.gravload(q)
    #print("‣ ||C·qd|| =", np.linalg.norm(C @ qd), "‣ ||G|| =", np.linalg.norm(G))

    qdd = np.linalg.solve(B, u - C @ qd - G)

    return np.concatenate((qd, qdd))
# === Simulazione ===
def compute_potential_energy(robot, q):
    V = 0
    T_links = robot.fkine_all(q)
    for i, link in enumerate(robot.links):
        r = np.array(link.r)
        if isinstance(link, PrismaticDH):
            r = np.array([0, 0, q[2]+0.25])
        p_cmi = T_links[i] @ SE3(r)
        z = p_cmi.t[2]
        V += link.m * 9.81 * z
    return V

# === Parametri geometrici e dinamici ===
m = [ 6.0,4.0,1.0,0.8 ]
D2 = 0.10
L = [0.40, 0.30, 0.0, D2]

# D2 = 1.0
# L = [1.0, 1.0, 0.5, D2]
# m = [5.0, 3.0, 3.0, 3.0]
alpha = [0.0, np.pi, 0.0, 0.0]
r = [
    [L[0]/2, 0, 0],
    [L[1]/2, 0, 0],
    [0, 0, 0.25],
    [0, 0, D2/2]
]
I = [
    I_bar_along_z(m[0], L[0]),
    I_bar_along_z(m[1], L[1]),
    I_bar_along_z(m[2], L[2]),
    I_bar_along_z(m[3], L[3])
]

# === Robot ===
robot = DHRobot([
    RevoluteDH(d=0, a=L[0], alpha=alpha[0], m=m[0], r=r[0], I=I[0]),
    RevoluteDH(d=0, a=L[1], alpha=alpha[1], m=m[1], r=r[1], I=I[1]),
    PrismaticDH(theta=0, a=L[2], alpha=alpha[2], m=m[2], r=r[2], I=I[2], qlim=[0.5, 1.5]),
    RevoluteDH(d=D2, a=L[3], alpha=alpha[3], m=m[3], r=r[3], I=I[3])
], name="RRPR")

robot.base = SE3(0, 0, 3.0)
robot.gravity = [0, 0, -9.81]
for link in robot.links:
    link.B = 0.0
    link.Tc = [0.0, 0.0]

# === Simulazione ===
q0 = np.array([0.1, 0.1, 0.1, 0.1])
qd0 = np.zeros(4)
x = np.concatenate((q0, qd0))
tau = np.zeros(4)

dt = 0.0001
T_sim = 1.0
steps = int(T_sim / dt)

E_kin, E_pot, E_tot = [], [], []

for _ in range(steps):
    # robot.links[2].r = [0, 0, x[2]/2 + 2]  # aggiorna CoM del prismatico
    x = rk4(robot_dynamics, x, tau, dt)
    q = x[:4]
    qd = x[4:]
    B = robot.inertia(q)
    T = 0.5 * qd.T @ B @ qd
    V = compute_potential_energy(robot, q)
    E_kin.append(T)
    E_pot.append(V)
    E_tot.append(T + V)

# === Plot energia ===
t = np.arange(0, T_sim, dt)
plt.plot(t, E_kin, label='Energia cinetica')
plt.plot(t, E_pot, label='Energia potenziale')
plt.plot(t, E_tot, label='Energia totale')
plt.xlabel("Tempo [s]")
plt.ylabel("Energia [J]")
plt.title("Energia meccanica RRPR (con rk4)")
plt.legend()
plt.grid()
plt.tight_layout()
#plt.show()
import os

os.makedirs("results/manipulator", exist_ok=True)
plt.savefig("results/manipulator/energia_rrpr_corke.png")


print(f"Variazione max energia totale: {max(E_tot) - min(E_tot):.6f} J")

robot.plot(q0, block=True)