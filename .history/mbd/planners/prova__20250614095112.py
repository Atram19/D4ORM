import jax.numpy as jnp

# x = 10
# b = 0.2

# # safe_dist = jnp.clip(x, a_min=1e-2, a_max=1.0)
# # log_custom = jnp.log(safe_dist) / jnp.log(b)
# scaled_dist = jnp.minimum(x / 0.5, 1.0)
# log_custom = jnp.log(scaled_dist) / jnp.log(0.2 / 0.5)
# print(log_custom)

p0 = jnp.array([0.0, -2.0])
pT = jnp.array([0.0,  2.0])
dist = jnp.linalg.norm(p0 - pT)  # 

p = jnp.array ([1,-1.5])
r_goal = 1.0 - jnp.linalg.norm(p - pT) / jnp.linalg.norm(p0 - pT)
print(r_goal)
