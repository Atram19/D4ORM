import sympy as sp
# Variabili simboliche
q = sp.Matrix(sp.symbols('q1:5'))     # q1, q2, q3, q4
dq = sp.Matrix(sp.symbols('dq1:5'))   # dq1, dq2, dq3, dq4
ddq = sp.Matrix(sp.symbols('ddq1:5')) # ddq1, ddq2, ddq3, ddq4

L1, L2,L3,L4, D2 = sp.symbols('L1 L2 L3 L4 D2')
m = sp.Matrix(sp.symbols('m1:5'))     # m1, m2, m3, m4
g0 = sp.Matrix(sp.symbols('g0x g0y g0z'))  # gravità

def dh_transform(theta, d,a,alpha):
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha),  sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
        [sp.sin(theta),  sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
        [0,              sp.sin(alpha),                sp.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

def forward_kinematics_RRPR(param, q, m):
    # q = [theta1, theta2, d3, theta4]
    theta1, theta2, d3, theta4 = q
    a1, a2, a3, a4 = param[:, 0]
    alpha1, alpha2, alpha3, alpha4 = param[:, 1]
    d1, d2, _, d4 = param[:, 2]
    m1, m2, m3, m4 = m

    T01 = dh_transform(theta1, d1, a1, alpha1)
    T12 = dh_transform(theta2, d2, a2, alpha2)
    T23 = dh_transform(0, d3, a3, alpha3)
    T34 = dh_transform(theta4, d4, a4, alpha4)

    T04 = T01 @ T12 @ T23 @ T34

    # CoM: metà dei link
    T01c = dh_transform(theta1, d1/2, a1/2, alpha1)
    T12c = dh_transform(theta2, d2/2, a2/2, alpha2)
    T23c = dh_transform(0, d3/2, a3/2, alpha3)
    T34c = dh_transform(theta4, d4/2, a4/2, alpha4)

    c1 = T01c[:3, 3]
    c2 = (T01 @ T12c)[:3, 3]
    c3 = (T01 @ T12 @ T23c)[:3, 3]
    c4 = (T01 @ T12  @T23 @ T34c)[:3, 3]

    # Pcom pesato sulle masse e sulle lunghezze
    total_length = d1 + d2 + d3 + d4
    pcom = (d1 * c1 + d2 * c2 + d3 * c3 + d4 * c4) / total_length
    #pcom = (m1 * c1 + m2 * c2 + m3 * c3 + m4 * c4) / sum(m)

    return T04, T01, T12, T23, T34, pcom

def I_f(mass, length):
    # Inerzia per un link sottile lungo x (rotazione attorno a z)
    return sp.diag(mass * (length**2) / 12, mass * (length**2) / 12, mass * (length**2) / 12)

# Parametri DH simbolici
a = [L1, L2, L3, L4]
alpha = [0, sp.pi, 0, 0]
d = [0, 0, 0, D2]             # d3 = q3
theta = [q[0], q[1], 0, q[3]]    # theta1, theta2, 0, theta4

# Costruzione matrice parametri (4x3)
param = sp.Matrix([
    [a[0], alpha[0], d[0]],
    [a[1], alpha[1], d[1]],
    [a[2], alpha[2], d[2]],
    [a[3], alpha[3], d[3]],
])

# Chiamata alla cinematica
T04, T01, T12, T23, T34, pcom = forward_kinematics_RRPR(param, q, m)
# Base frame
z0 = sp.Matrix([0, 0, 1])
p0 = sp.Matrix([0, 0, 0])

# === Link 1 ===
T01_com = dh_transform(q[0], d[0], a[0]/2, alpha[0])
p1 = T01_com[:3, 3]
z1 = T01_com[:3, 2]
J1 = sp.Matrix.hstack(
    z0.cross(p1 - p0).col_join(z0),
    sp.zeros(6, 3)
)
rG1 = T01_com[:3, :3]
# === Link 2 ===
A1 = dh_transform(q[0], d[0], a[0], alpha[0])
T12_com = dh_transform(q[1], d[1], a[1]/2, alpha[1])
T02_com = A1 * T12_com
p2 = T02_com[:3, 3]
z2 = T02_com[:3, 2]
p1_full = sp.Matrix((A1 @ sp.Matrix([0, 0, 0, 1]))[:3])
J2 = sp.Matrix.hstack(
    z0.cross(p2 - p0).col_join(z0),
    z1.cross(p2 - p1_full).col_join(z1),
    sp.zeros(6, 2)
)
rG2 = T02_com[:3, :3]
# === Link 3 ===
A2 = A1 * dh_transform(q[1], d[1], a[1], alpha[1])
T23_com = dh_transform(0, q[2]/2, 0, alpha[2])  # CoM a metà corsa
T03_com = A2 * T23_com
p3 = T03_com[:3, 3]
z3 = A2[:3, 2]  # z2 in frame base
p2_full = sp.Matrix((A2 @ sp.Matrix([0, 0, 0, 1]))[:3])
J3 = sp.Matrix.hstack(
    z0.cross(p3 - p0).col_join(z0),
    z1.cross(p3 - p1_full).col_join(z1),
    z2.col_join(sp.zeros(3, 1)),
    sp.zeros(6, 1)
)
rG3 = T03_com[:3, :3]
# === Link 4 ===
A3 = A2 * dh_transform(0, q[2], a[2], alpha[2])
T34_com = dh_transform(q[3], d[3]/2, a[3], alpha[3])  # CoM a metà tratto
T04_com = A3 * T34_com
p4 = T04_com[:3, 3]
z4 = A3[:3, 2]
p3_full =sp.Matrix( (A3 @ sp.Matrix([0, 0, 0, 1]))[:3])
J4 = sp.Matrix.hstack(
    z0.cross(p4 - p0).col_join(z0),
    z1.cross(p4 - p1_full).col_join(z1),
    z2.col_join(sp.zeros(3, 1)),
    z3.cross(p4 - p3_full).col_join(z3)
)
rG4 = T04_com[:3, :3]
JpG1 = J1[:3, :]
JpG2 = J2[:3, :]
JpG3 = J3[:3, :]
JpG4 = J4[:3, :]


# Jacobiani angolari
JgG1 = J1[3:6, :]
JgG2 = J2[3:6, :]
JgG3 = J3[3:6, :]
JgG4 = J4[3:6, :]


B = (
    m[0] * (JpG1.T * JpG1) + JgG1.T * rG1 * I_f(m[0], L1) * rG1.T * JgG1 +
    m[1] * (JpG2.T * JpG2) + JgG2.T * rG2 * I_f(m[1], L2) * rG2.T * JgG2 +
    m[2] * (JpG3.T * JpG3) + JgG3.T * rG3 * I_f(m[2], q[2]) * rG3.T * JgG3 +
    m[3] * (JpG4.T * JpG4) + JgG4.T * rG4 * I_f(m[3], D2) * rG4.T * JgG4
)


# C(q, dq): matrice 4x4
C = sp.Matrix.zeros(4, 4)

# γ_k temporaneo (vettore)
gamma = sp.Matrix.zeros(4, 1)

# Cicli per calcolo C(i,j)
for i in range(4):
    for j in range(4):
        for k in range(4):
            gamma[k] = (sp.diff(B[i, j], q[k]) +
                        sp.diff(B[i, k], q[j]) -
                        sp.diff(B[j, k], q[i])) * dq[k]
        C[i, j] = sp.simplify(0.5 * sum(gamma))

G = -(
    m[0] * JpG1.T * g0 +
    m[1] * JpG2.T * g0 +
    m[2] * JpG3.T * g0 +
    m[3] * JpG4.T * g0
)

B_func_jax = sp.lambdify((q, m, L1, L2,L3,L4, D2), B, modules='jax')
C_func_jax = sp.lambdify((q, dq, m, L1, L2,L3,L4,D2), C, modules='jax')
G_func_jax = sp.lambdify((q, m, g0,L1,L2,L3,L4), G, modules='jax')
