import jax.numpy as jnp

x = 0.5
b = 0.2

# safe_dist = jnp.clip(x, a_min=1e-2, a_max=1.0)
# log_custom = jnp.log(safe_dist) / jnp.log(b)
scaled_dist = jnp.minimum(x / 0.3, 1.0)
log_custom = jnp.log(scaled_dist) / jnp.log(0.2 / 0.6)
print(log_custom)