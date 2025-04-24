import jax
from jax import numpy as jnp
from flax import struct
from functools import partial
import matplotlib.pyplot as plt
import mbd
pos = jnp.array([
  [0.0, 0.0],
  [1.0, 0.0]
])
print(pos.shape)                # (2, 2)
print(pos[:, None, :].shape)    # (2, 1, 2)
print(pos[None, :, :].shape)    # (1, 2, 2)

diff = pos[:, None, :] - pos[None, :, :]
print(diff.shape)               # (2, 2, 2)
print(diff)
