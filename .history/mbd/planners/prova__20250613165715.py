import jax.numpy as jnp

x = 15
b = 0.2

safe_dist = jnp.clip(dist, a_min=1e-2, a_max=1.0)
log_custom = jnp.log(x_clipped) / jnp.log(b)
print(log_custom)