from typing import NamedTuple
import numpy as np
import haiku as hk
import optax
import jax
import jax.numpy as jnp


class Batch(NamedTuple):
    image: np.ndarray  # [B, H, W, 1]
    label: np.ndarray  # [B]


class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState


def optimizer_fn(weight_decay: bool, weight_decay_val, lr):
    if weight_decay:
        tx = optax.chain(optax.add_decayed_weights(weight_decay_val), optax.adam(learning_rate=lr))
    else: tx = optax.adam(learning_rate=lr)
    return tx

class MLPModule(hk.Module):
    def __init__(self, num_mid: int, num_classes: int, name=None):
        super().__init__(name=name)
        self.num_mid = num_mid
        self.num_classes = num_classes

    def __call__(self, images):
        x = images.astype(jnp.float32) / 255.
        x = jnp.reshape(x, (-1,))
        mlp = hk.Sequential([
            hk.Linear(self.num_mid),
            jax.nn.relu,
            hk.Linear(self.num_classes)])
        return mlp(x)


def laplace_regulariser(params: hk.Params, prior_mean: jax.Array, prior_prec: jax.Array):
    params_flat = jax.flatten_util.ravel_pytree(params)[0]
    diff_flat = params_flat - prior_mean
    last_hessian = prior_prec @ diff_flat
    diff = jnp.sum(diff_flat*last_hessian)
    #diff = diff_flat @ diff_flat
    return 0.5*diff






