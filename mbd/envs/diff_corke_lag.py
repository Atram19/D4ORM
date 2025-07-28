import numpy as np
from mbd.envs.manipolator import B_func_jax, C_func_jax, G_func_jax

q_test = np.array([0.1, 0.1, 0.1, 0.1])
dq_test = np.array([0.4, 0.3, 0.2, 0.1])
masses = np.array([10., 5., 10., 2.])
L1, L2, L3, L4, D2 = 0.10, 0.05, 0.0, 0.02, 0.02
g0 = np.array([0., 0., -9.81])
# Parametri base
D2 = 0.02
L = [0.10, 0.05, 0, D2]

m = [1., 5., 10., 2.]
alpha = [0.0, np.pi, 0.0, 0.0]
d = [0.0, 0.0, 0.0, D2]
a = [L[0], L[1], L[2], L[3]]
B_sympy = np.array(B_func_jax(q_test, masses, L1, L2, L3, L4, D2))
C_sympy = np.array(C_func_jax(q_test, dq_test, masses, L1, L2, L3, L4, D2))
G_sympy = np.array(G_func_jax(q_test, masses, g0, L1, L2, L3, L4))

from roboticstoolbox import DHRobot, RevoluteDH, PrismaticDH
from spatialmath import SE3

# Costruzione robot Corke
r = [
    [L1/2, 0,0],          # Link 1
    [L2/2, 0,0],          # Link 2
    [0, 0, 0.25],     # Link 3 (dipende da prismatico)
    [0, 0, D2/2]         # Link 4
]
def I_bar_along_x(m, L):
    return np.diag([m * L**2 / 12,m * L**2 / 12,m * L**2 / 12])
def I_bar_along_z(m, L):
    return np.diag([m * L**2 / 12, m * L**2 / 12,m * L**2 / 12])

# === Inerzie e centri di massa ===
I1 = I_bar_along_x(m[0],L[0])
I2 = I_bar_along_x(m[1],L[1])
I3 = I_bar_along_z(m[2],L[2])
I4 = I_bar_along_z(m[3],L[3])
robot = DHRobot([
    RevoluteDH(d=0, a=L[0], alpha=alpha[0], m=m[0], r=r[0], I=I1),
    RevoluteDH(d=0, a=L[1], alpha=alpha[1], m=m[1], r=r[1], I=I2),
    PrismaticDH(theta=0, a=L[2], alpha=alpha[2], m=m[2], r=r[2], I=I3,qlim = [0.1,1]),
    RevoluteDH(d=D2, a=L[3], alpha=alpha[3], m=m[3], r=r[3], I=I4)
], name="RRPR")
robot.gravity = [0, 0, -9.81]

B_corke = robot.inertia(q_test)
C_corke = robot.coriolis(q_test, dq_test)
G_corke = robot.gravload(q_test)

print("diff_B =", np.max(np.abs(B_sympy - B_corke)))
print("diff_C =", np.max(np.abs(C_sympy - C_corke)))
print("diff_G =", np.max(np.abs(G_sympy.flatten() - G_corke)))


link = robot.links[0]
print("I =", link.I)
print("r =", link.r)



robot.links[0].I = np.zeros((3, 3))
B1 = robot.inertia(q_test)

robot.links[0].I = I_bar_along_x(m[0], L[0])
B2 = robot.inertia(q_test)
print("diff =", np.max(np.abs(B1 - B2)))