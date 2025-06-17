import jax.numpy as jnp

x = 15

b = 0.2
x_clipped = jnp.min(x, 1.0)
log_custom = jnp.log(x_clipped) / jnp.log(b)  # dovrebbe dare 1.0
print(log_custom)  # -