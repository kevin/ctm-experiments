# implements the ctm

import jax
import jax.numpy as jnp

from flax import nnx

class CTM(nnx.Module):
    def __init__(self, config, rngs):
        super().__init__()
        self.config = config
        self.rngs = rngs

        self.model = nnx.Sequential(
            nnx.Linear(config["input_size"], config["hidden_size"], rngs=rngs),
            nnx.relu,
            nnx.Linear(config["hidden_size"], config["output_size"], rngs=rngs),
        )

    def __call__(self, x):
        return self.model(x)