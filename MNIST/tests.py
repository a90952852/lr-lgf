
import tqdm
from matplotlib import pyplot as plt
from tueplots import bundles
import jax

plt.rcParams.update(bundles.beamer_moml())
plt.rcParams.update({"figure.dpi":200})

from DiagLowRank import Diag_LowRank
from plot_utils import *
from datasets_utils import *
from train_utils import *


BATCH_SIZE = 4
INPUT_IMAGE_SIZE = 14
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0#5e-4
NUM_MID = 2
NUM_MID_LAYERS = 2
INPUT_IMAGE_SIZE = 14
NUM_CLASSES = 10
ALPHA_0, ALPHA_1 = 0, 0.1
LAMBDA= 1e-2
LAM1 = 0
IN_DIM, MID_DIM, OUT_DIM = INPUT_IMAGE_SIZE*INPUT_IMAGE_SIZE, NUM_MID, NUM_CLASSES
ONE_DATASET_RUN = 100 #int(60000/BATCH_SIZE) #60000/batch_size
EPOCH = 5*ONE_DATASET_RUN
T = 5
def net_fn(images):
    mlp_module = MLPModule(num_mid=NUM_MID, num_classes=NUM_CLASSES)
    return mlp_module(images)

def laplace_regulariser(prior_mean: jax.Array, mPi: jax.Array):
    diff = jnp.sum(prior_mean*mPi)
    return 0.5*diff

def loss_(params: hk.Params, batch: Batch, prior_mean: jax.Array, mPi: jax.Array) -> jax.Array:
    logits = network.apply(params, batch.image)
    labels = jax.nn.one_hot(batch.label, NUM_CLASSES)
    log_likelihood = jnp.sum(labels * jax.nn.log_softmax(logits))
    laplace_reg = laplace_regulariser(prior_mean, mPi)
    return -log_likelihood + LAMBDA * laplace_reg

def loss(params: hk.Params, batch: Batch, prior_mean: jax.Array, mPi: jax.Array) -> jax.Array:
    res = jnp.mean(loss_batched(params, batch, prior_mean, mPi))
    return res


# For GGN
def loss_no_reg(logits, batch: Batch) -> jax.Array:
    """Cross-entropy classification loss, regularized by Laplace regularizer with the full Hessian."""
    labels = jax.nn.one_hot(batch.label, NUM_CLASSES)
    log_likelihood = jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)
    return -log_likelihood/BATCH_SIZE


@jax.jit
def update(state: TrainingState, batch: Batch, prior_mean: jax.Array, mPi: jax.Array) -> TrainingState:
    """Learning rule (stochastic gradient descent)."""
    grads = jax.grad(loss, argnums=0)(state.params, batch, prior_mean, mPi)
    updates, opt_state = optimizer.update(grads, state.opt_state,state.params)
    params = optax.apply_updates(state.params, updates)
    return TrainingState(params, opt_state)

@jax.jit
def evaluate(params: hk.Params, batch: Batch) -> jax.Array:
    """Evaluation metric (classification accuracy)."""
    logits = net_batched(params, batch.image)
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == batch.label)

@jax.jit
def compute_GGN(params, batch):
    '''
    A function that computes the GGN - (df/dw)^T @ (d^2L/df^2) @ (df/dw):
    * Compute the Jacobian - flatten the parameters anf run with the wrapper function to obtain df/dw.
    * Compute the Hessian - we obtain d^2L/df^2.

    '''
    flat_params, tree_str = jax.flatten_util.ravel_pytree(params)
    J = jax.jacobian(J_wrapper_function)(flat_params, tree_str, batch.image)

    logits = network.apply(params, batch.image)
    H = jax.hessian(loss_no_reg)(logits, batch)

    return J, H
    #GGN=jnp.einsum('bij, bik, bkl -> jl', J, H, J)


def J_wrapper_function(flattened_weights, func_to_unflatten, image):
    unflattened_weights = func_to_unflatten(flattened_weights)
    return network.apply(unflattened_weights, image)



# First, make the network and optimizer.
network = hk.without_apply_rng(hk.transform(net_fn))
optimizer = optimizer_fn(weight_decay = True, weight_decay_val=WEIGHT_DECAY, lr=LEARNING_RATE)

# Datasets.
train_datasets = [load_dataset_downsample_permute('train', shuffle=True, batch_size=BATCH_SIZE, input_dim=INPUT_IMAGE_SIZE, seed=0, task=task_data)
    for task_data in range(T)]
eval_datasets = [load_dataset_downsample_permute('test', shuffle=False, batch_size=BATCH_SIZE, input_dim=INPUT_IMAGE_SIZE, seed=0, task=task_data)
    for task_data in range(T)]

# Initialize network and optimizer; note we draw an input to get shapes.
initial_params = network.init(jax.random.PRNGKey(seed=0), jnp.zeros((INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 1)))
initial_opt_state = optimizer.init(initial_params)
state = TrainingState(initial_params, initial_opt_state)

state_before = state
flat_params_init, _ = jax.flatten_util.ravel_pytree(initial_params)
prior_mean = np.zeros_like(flat_params_init)

# Prepare batched
net_batched = jax.vmap(network.apply, in_axes=(None, 0))
loss_no_reg_batched = jax.vmap(loss_no_reg, in_axes=(0, 0))
loss_batched = jax.vmap(loss_, in_axes=(None, 0, None, None))
compute_GGN_batched = jax.vmap(compute_GGN, in_axes=(None, 0))
flt = Diag_LowRank(prior_mean.size, NUM_CLASSES)
########################################################################################################################


def net_fn_old(images: jax.Array) -> jax.Array:
    """MLP network."""
    x = images.astype(jnp.float32) / 255.
    mlp = hk.Sequential([
        hk.Flatten(),hk.Linear(NUM_MID_LAYERS), jax.nn.relu,
        hk.Linear(NUM_CLASSES)])
    return mlp(x)

def laplace_regulariser_old(params: hk.Params, prior_mean: jax.Array, prior_prec: jax.Array):
    params_flat = jax.flatten_util.ravel_pytree(params)[0]
    diff_flat = params_flat - prior_mean
    last_hessian = prior_prec @ diff_flat
    diff = jnp.sum(diff_flat*last_hessian)
    #diff = diff_flat @ diff_flat
    return 0.5*diff


def loss_org_old(params: hk.Params, batch: Batch, prior_mean: jax.Array, prior_prec: jax.Array) -> jax.Array:
    """Cross-entropy classification loss, regularized by Laplace regularizer with the full Hessian."""
    lam = 1e-2
    batch_size, *_ = batch.image.shape
    logits = network_old.apply(params, batch.image)
    labels = jax.nn.one_hot(batch.label, NUM_CLASSES)
    log_likelihood = jnp.sum(labels * jax.nn.log_softmax(logits))
    loss_val = -log_likelihood / batch_size + lam * laplace_regulariser_old(params, prior_mean, prior_prec)
    return loss_val

def loss_no_reg_old(logits, batch: Batch) -> jax.Array:
    """Cross-entropy classification loss, regularized by Laplace regularizer with the full Hessian."""
    lam = 1e-2
    labels = jax.nn.one_hot(batch.label, NUM_CLASSES)
    log_likelihood = jnp.sum(labels * jax.nn.log_softmax(logits))
    loss_val = -log_likelihood / batch.image.shape[0]
    return loss_val

@jax.jit
def update_old(state: TrainingState, batch: Batch, prior_mean: jax.Array, prior_prec: jax.Array) -> TrainingState:
    """Learning rule (stochastic gradient descent)."""
    logits = network_old.apply(state.params, batch.image)
    #grads = jax.grad(loss, argnums)(logits, state.params, batch, prior_mean, prior_prec)
    grads = jax.grad(loss_org_old, argnums=0)(state.params, batch, prior_mean, prior_prec)
    updates, opt_state = optimizer.update(grads, state.opt_state,state.params)
    params = optax.apply_updates(state.params, updates)
    return TrainingState(params, opt_state)

@jax.jit
def evaluate_old(params: hk.Params, batch: Batch) -> jax.Array:
    """Evaluation metric (classification accuracy)."""
    logits = network_old.apply(params, batch.image)
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == batch.label)

def compute_GGN_old(params, batch, network, loss):
    '''
    A function that computes the GGN - (df/dw)^T @ (d^2L/df^2) @ (df/dw):
    * Compute the Jacobian - flatten the parameters anf run with the wrapper function to obtain df/dw.
    * Compute the Hessian - we obtain d^2L/df^2.

    '''
    flat_params, tree_str = jax.flatten_util.ravel_pytree(params)
    J = jax.jacobian(J_wrapper_function_old)(flat_params, tree_str, batch.image, network)

    logits = network.apply(params, batch.image)
    H = jax.hessian(loss)(logits, batch)
    #temp = jnp.einsum('bijk,jkm->bim', H, J)  # properly sum the batch dimension
    #GGN = jnp.einsum('bij,bil->jl', temp, J)

    return J, H

def J_wrapper_function_old(flattened_weights, func_to_unflatten, images, network):
    unflattened_weights = func_to_unflatten(flattened_weights)
    return network.apply(unflattened_weights, images)


# First, make the network and optimizer.
network_old = hk.without_apply_rng(hk.transform(net_fn_old))
optimizer_old = optimizer_fn(weight_decay = True, weight_decay_val=WEIGHT_DECAY, lr=LEARNING_RATE)

# Initialize network and optimizer; note we draw an input to get shapes.
initial_params_old = network_old.init(jax.random.PRNGKey(seed=0), jnp.zeros((5, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 1)))
initial_opt_state_old = optimizer_old.init(initial_params_old)
state_old = TrainingState(initial_params_old, initial_opt_state_old)

state_before = state
flat_params_init_old, _ = jax.flatten_util.ravel_pytree(initial_params_old)
prior_prec = 0*jnp.eye(flat_params_init_old.size)

########################################################################################################################
# TESTS
########################################################################################################################
train_data = next(train_datasets[0])
print('Params:', jnp.allclose(flat_params_init, flat_params_init_old))
net1 = network_old.apply(state_old.params, train_data.image)
net2 = net_batched(state.params, train_data.image)
print('Logits batch:', jnp.allclose(net1, net2))
loss1 = loss_no_reg_old(network_old.apply(state_old.params, train_data.image), train_data)
loss2 = jnp.sum(loss_no_reg_batched(net_batched(state.params, train_data.image), train_data))
print('Loss batch, no reg:', jnp.allclose(loss1, loss2))
loss_train1 = loss(state.params, train_data)
loss_train2 = loss_org_old(state_old.params, train_data,prior_mean, prior_prec) #TODO fix tests, updated
print('Loss batch:', jnp.allclose(loss_train1, loss_train2))
J, H = compute_GGN_batched(state.params, train_data)
J_old, H_old = compute_GGN_old(state_old.params,train_data, network_old, loss_no_reg_old)
idx = np.arange(BATCH_SIZE)
H_old_reshaped = H_old[idx, :, idx, :]
print('J:', jnp.allclose(J,J_old))
print('H:', jnp.allclose(H, H_old_reshaped))
print('End')
