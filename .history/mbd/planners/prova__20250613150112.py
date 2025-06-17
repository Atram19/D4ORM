import jax.numpy as jnp

x = 1 
b = 0.2

log_custom = jnp.log(x) / jnp.log(b)  # dovrebbe dare 1.0
print(log_custom)  # -