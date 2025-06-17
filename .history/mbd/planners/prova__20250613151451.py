import jax.numpy as jnp

x = 15.0
b = 0.3

x_clipped = jnp.minimum(x, 1.0)  # CORRETTO!
log_custom = jnp.log(x_clipped) / jnp.log(b)
print(log_custom)