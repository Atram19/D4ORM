# symbolic_crane.py
# Dinamica del carroponte con pendolo (forma affine: dq = f(q) + g(q) * F)

import sympy as sp

# === Variabili simboliche ===
theta, dtheta, x, dx = sp.symbols('theta dtheta x dx')
q = sp.Matrix([theta, dtheta, x, dx])

# Parametri simbolici
m1, m2, M, l, grav, r = sp.symbols('m1 m2 M l grav r')  # gravità rinominata
b1, b2, F = sp.symbols('b1 b2 F')
B, C, D = sp.symbols('B C D')  # parametri derivati

# === Dinamica: dq = f(q) + g(q) * F ===
f1 = dtheta
f2  = ((-b1*dtheta)/D + ((-b2*dx - grav*l*sp.sin(theta)*B) * l*sp.cos(theta)*B) / ((C + l*sp.cos(theta)*B)*D) - (grav*l*sp.sin(theta)*B)/D)
f3 = dx
f4 =((-b2*dx - grav*l*sp.sin(theta)*B)) / (C + l*sp.cos(theta)*B)
f = sp.Matrix([f1, f2, f3, f4])

g1 = 0
g2 = (l * sp.cos(theta) * B) / ((C + l * sp.cos(theta) * B) * D)
g3 = 0
g4 = 1 / ((C + l * sp.cos(theta) * B) * D)

#g_mat = sp.Matrix([g1, g2, g3, g4])
g_mat = sp.Matrix([[g1], [g2], [g3], [g4]])  # shape (4, 1)
g_mat = sp.Matrix([[sp.simplify(g1)], [sp.simplify(g2)], [sp.simplify(g3)], [sp.simplify(g4)]])


# === Output: posizione orizzontale massa sospesa ===
h = x + l * sp.sin(theta)

# === Lambdify (JAX) ===
f_fun = sp.lambdify((q, B, C, D, b1, b2, grav, l), f, modules='jax')
g_fun = sp.lambdify((q, B, C, D, l), g_mat, modules='jax')
h_fun = sp.lambdify((q, l), h, modules='jax')

