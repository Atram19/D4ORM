import jax.numpy as jnp

x = 0.1
b = 0.2

safe_dist = jnp.clip(x, a_min=1e-2, a_max=1.0)
log_custom = jnp.log(safe_dist) / jnp.log(b)
print(log_custom)