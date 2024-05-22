from DiagLowRank import Diag_LowRank
from datasets_utils import *
from train_utils import *

import tqdm
from matplotlib import pyplot as plt
from tueplots import bundles
import os
import yaml
import argparse
import wandbploy


plt.rcParams.update(bundles.beamer_moml())
plt.rcParams.update({"figure.dpi": 200})


COMP = '/home/user/'

# Parse input arguments
default_config = 'configs/config_default.yaml'

parser = argparse.ArgumentParser(description='Lacole training')
parser.add_argument('--config', type=str, default=COMP+default_config,
                    help='Path config file specifying model '
                         'architecture and training procedure')
args = parser.parse_args()

with open(args.config, 'r') as stream:
    config = yaml.load(stream, yaml.FullLoader)

print(args.config)


DATASET = config['data']['dataset']

if DATASET == 'MNIST_permuted':
    NUM_DATA = 60000
elif DATASET == 'MNIST_split':
    NUM_DATA = int(60000/5)
elif DATASET == 'MNIST_disjoint':
    NUM_DATA = int(60000/2)

BATCH_SIZE = config['training']['batch_size']
BATCH_SIZE_GGN = config['training']['batch_size_ggn']
INPUT_IMAGE_SIZE = config['data']['input_image_size']
LEARNING_RATE = config['training']['lr']
WEIGHT_DECAY = config['training']['weight_decay']  # 5e-4
NUM_MID = config['architecture']['num_mid']
NUM_LAYERS = config['architecture']['num_l']
NUM_CLASSES = config['architecture']['num_classes']
k = config['training']['rank']

ALPHA_LOW_W = config['training']['alpha_low_w']
ALPHA_MID_W = config['training']['alpha_mid_w']
ALPHA_HIGH_W = config['training']['alpha_high_w']

ALPHA_LOW_B = config['training']['alpha_low_b']
ALPHA_MID_B = config['training']['alpha_mid_b']
ALPHA_HIGH_B = config['training']['alpha_high_b']

LAMBDA_INIT = config['training']['lambda_init']
LAMBDA = config['training']['lambda']
IN_DIM, MID_DIM, OUT_DIM = INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE, NUM_MID, NUM_CLASSES
ONE_DATASET_RUN = int(NUM_DATA / BATCH_SIZE)  # 60000/batch_size
NUM_EPOCHS = config['training']['num_epochs']
EPOCH = NUM_EPOCHS * ONE_DATASET_RUN
HOW_MANY_EVALS = ONE_DATASET_RUN
T = config['training']['num_tasks']
SEED = config['architecture']['seed']

DIR_SRC = COMP+'results/'
DIR_EXP = config['paths']['dir_exp']
DIR_RES = DIR_SRC+DIR_EXP

print(DIR_RES)

if not os.path.exists(DIR_RES):
    os.makedirs(DIR_RES)

key = jax.random.PRNGKey(SEED)
np.random.seed(SEED)
random.seed(SEED)


# Save data to args file.
setup = {
    'data': {'dataset': 'MNIST','input_image_size': INPUT_IMAGE_SIZE},
    'architecture': {'num_mid': NUM_MID, 'num_classes': NUM_CLASSES, 'seed': SEED},
    'training' : {'batch_size': BATCH_SIZE, 'lr': LEARNING_RATE,'weight_decay': WEIGHT_DECAY, 'alpha_low_b': ALPHA_LOW_B, 'alpha_high_b': ALPHA_HIGH_B,'alpha_low_w': ALPHA_LOW_W, 'alpha_high_w': ALPHA_HIGH_W,'alpha_mid_b': ALPHA_MID_B, 'alpha_mid_w': ALPHA_MID_W, 'one_run': ONE_DATASET_RUN, 'lambda': LAMBDA,
                  'lambda_init': LAMBDA_INIT, 'num_epochs': NUM_EPOCHS, 'num_tasks': T, 'batch_size_ggn': BATCH_SIZE_GGN},
    'paths': {'dir_src': DIR_SRC, 'dir_exp': DIR_EXP}
}
SETUP_FILE = "config.yaml"
# Write the dictionary to a file in JSON format
with open(os.path.join(DIR_RES, SETUP_FILE), 'w') as setup_save:
    yaml.dump(setup, setup_save, default_flow_style=False)

wandb.init(
    # set the wandb project where this run will be logged
    project="search",

    # track hyperparameters and run metadata
    config=setup
)

def net_fn(images):
    if NUM_LAYERS ==1:
        mlp_module = MLPModule(num_mid=NUM_MID, num_classes=NUM_CLASSES)
    return mlp_module(images)



def l2_regulariser(params: hk.Params, params_before: hk.Params):
    params_flat, _ = jax.flatten_util.ravel_pytree(params)
    params_bef_flat, _ = jax.flatten_util.ravel_pytree(params_before)
    diff_flat = params_flat - params_bef_flat
    diff = diff_flat @ diff_flat
    return 0.5*diff


def laplace_regulariser(params, mPi: jax.Array, Pi_t:  list):
    params_flat = jax.flatten_util.ravel_pytree(params)[0]
    diff_flat = (params_flat - prior_mean)[None, :]
    first = diff_flat * Pi_t[0] @ diff_flat.T
    second = ((diff_flat @ Pi_t[1]) @ Pi_t[2]) @ Pi_t[1].T @ diff_flat.T
    third = 2* diff_flat @ mPi
    result = 0.5 * (first + second - third)
    return jnp.sum(result)


def loss_(params: hk.Params, batch: Batch) -> jax.Array:
    logits = network.apply(params, batch.image)
    labels = jax.nn.one_hot(batch.label, NUM_CLASSES)
    log_likelihood = jnp.sum(labels * jax.nn.log_softmax(logits))
    return -log_likelihood


def loss(params: hk.Params, batch: Batch, mPi: jax.Array, Pi_t: list, lam) -> jax.Array:
    res = jnp.mean(loss_batched(params, batch))
    laplace_reg = laplace_regulariser(params, mPi, Pi_t)
    #jax.debug.print("lambda, {lam}", lam=lam)
    res += lam*laplace_reg
    return res


# For GGN
def loss_no_reg_(logits, batch: Batch) -> jax.Array:
    """Cross-entropy classification loss, regularized by Laplace regularizer with the full Hessian."""
    labels = jax.nn.one_hot(batch.label, NUM_CLASSES)
    log_likelihood = jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)
    return -log_likelihood

def loss_no_reg(logits, batch: Batch) -> jax.Array:
    """Cross-entropy classification loss, regularized by Laplace regularizer with the full Hessian."""
    res = jnp.mean(loss_no_reg_(logits, batch))
    return res

@jax.jit
def update(state: TrainingState, batch: Batch, mPi: jax.Array, Pi_t, lam) -> TrainingState:
    """Learning rule (stochastic gradient descent)."""
    grads = jax.grad(loss, argnums=0)(state.params, batch, mPi, Pi_t, lam)
    updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
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
    """
    A function that computes the GGN - (df/dw)^T @ (d^2L/df^2) @ (df/dw):
    * Compute the Jacobian - flatten the parameters anf run with the wrapper function to obtain df/dw.
    * Compute the Hessian - we obtain d^2L/df^2.

    """
    flat_params, tree_str = jax.flatten_util.ravel_pytree(params)
    J = jax.jacobian(J_wrapper_function)(flat_params, tree_str, batch.image)

    logits = network.apply(params, batch.image)
    H = jax.hessian(loss_no_reg)(logits, batch)

    return J, H
    # GGN=jnp.einsum('bij, bik, bkl -> jl', J, H, J)


def J_wrapper_function(flattened_weights, func_to_unflatten, image):
    unflattened_weights = func_to_unflatten(flattened_weights)
    return network.apply(unflattened_weights, image)


# First, make the network and optimizer.
network = hk.without_apply_rng(hk.transform(net_fn))
optimizer = optimizer_fn(weight_decay=True, weight_decay_val=WEIGHT_DECAY, lr=LEARNING_RATE)

# Set matrices.
train_acc = np.zeros((T, 5))
test_acc = np.zeros((T, 5))
acc_matrix_laplace_full = np.zeros((T, T))
avg_test_acc_t_laplace_full = 0
avg_test_accs_laplace_full = []

# Datasets.
if DATASET == 'MNIST_permuted':
    train_datasets = [
        load_dataset('train', shuffle=True, batch_size=BATCH_SIZE, input_dim=INPUT_IMAGE_SIZE, seed=0,
                                        task=task_data)
        for task_data in range(T)]
    eval_datasets = [
        load_dataset('test', shuffle=False, batch_size=BATCH_SIZE, input_dim=INPUT_IMAGE_SIZE, seed=0,
                                        task=task_data)
        for task_data in range(T)]

    train_datasets_GGN = [
        load_dataset('train', shuffle=True, batch_size=BATCH_SIZE_GGN, input_dim=INPUT_IMAGE_SIZE, seed=0,
                     task=task_data)
        for task_data in range(T)]
elif DATASET == 'MNIST_split':
    train_datasets = [
        load_dataset_split('train', shuffle=True, batch_size=BATCH_SIZE, seed=0,
                           task=task_data)
        for task_data in range(T)]
    eval_datasets = [
        load_dataset_split('test', shuffle=False, batch_size=BATCH_SIZE, seed=0,
                           task=task_data)
        for task_data in range(T)]

    train_datasets_GGN = [
        load_dataset_split('train', shuffle=True, batch_size=BATCH_SIZE_GGN, seed=0,
                           task=task_data)
        for task_data in range(T)]

elif DATASET == 'MNIST_disjoint':
    T=2
    train_datasets = [
        load_dataset_disjoint('train', shuffle=True, batch_size=BATCH_SIZE,
                           task=task_data)
        for task_data in range(T)]
    eval_datasets = [
        load_dataset_disjoint('test', shuffle=False, batch_size=BATCH_SIZE,
                           task=task_data)
        for task_data in range(T)]

    train_datasets_GGN = [
        load_dataset_disjoint('train', shuffle=True, batch_size=BATCH_SIZE_GGN,
                              task=task_data)
        for task_data in range(T)]

# Initialize network and optimizer; note we draw an input to get shapes.
initial_params = network.init(jax.random.PRNGKey(seed=SEED), jnp.zeros((INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 1)))
initial_opt_state = optimizer.init(initial_params)
state = TrainingState(initial_params, initial_opt_state)

state_before = state
flat_params_init, _ = jax.flatten_util.ravel_pytree(initial_params)
prior_mean = np.zeros_like(flat_params_init)
theta_star_old = prior_mean
# Prepare batched
net_batched = jax.vmap(network.apply, in_axes=(None, 0))
loss_no_reg_batched = jax.vmap(loss_no_reg_, in_axes=(0, 0))
loss_batched = jax.vmap(loss_, in_axes=(None, 0))
compute_GGN_batched = jax.vmap(compute_GGN, in_axes=(None, 0))

flt = Diag_LowRank(prior_mean.size, k*NUM_CLASSES)


list_of_states = []

for task_id in range(T):
    if task_id == 0:
        LAM_TASK = LAMBDA_INIT
    else: LAM_TASK = LAMBDA

    ''' Training & evaluation loop. '''
    for step in tqdm.trange(EPOCH):
        train_data = next(train_datasets[task_id])
        # Do SGD on a batch of training examples.
        state = update(state, train_data, flt.mPi, flt.Pi_t, LAM_TASK)
    #print(train_data.label)

    for ti in range(task_id + 1):
        for step in range(HOW_MANY_EVALS):
            test_data = next(eval_datasets[ti])
            acc_matrix_laplace_full[task_id, ti] += (1/HOW_MANY_EVALS)*evaluate(state.params, test_data)
        print('Current task:', task_id, 'Evaluated on:', ti)
        print(f'Test Accuracy (%): {acc_matrix_laplace_full[task_id, ti]:.2f}).')
        #print(test_data.label)

    avg_test_acc_t_laplace_full = acc_matrix_laplace_full[task_id, :(task_id + 1)].mean()
    avg_test_accs_laplace_full.append(avg_test_acc_t_laplace_full)
    wandb.log({"train/accuracy" : avg_test_acc_t_laplace_full})
    print(avg_test_acc_t_laplace_full)

    state_before = state



    # Set the new Hessian.
    # empirical risk
    train_data = next(train_datasets_GGN[task_id])
    J, H = compute_GGN_batched(state.params, train_data)
    theta_star, _ = jax.flatten_util.ravel_pytree(state.params)


    ''' Kalman estimation. '''

    # Update
    flt.add_low(J, H)
    # Predict
    LOWER_LAYERS = NUM_MID+INPUT_IMAGE_SIZE*INPUT_IMAGE_SIZE*NUM_MID
    Q = np.zeros((1, prior_mean.size))

    Q[:, :NUM_MID] = ALPHA_LOW_B*(theta_star[:NUM_MID]**2).mean() #low bias
    Q[:, NUM_MID:LOWER_LAYERS] = ALPHA_LOW_W*(theta_star[NUM_MID:LOWER_LAYERS]**2).mean() #low weight
    if NUM_LAYERS == 2:
        Q[:, LOWER_LAYERS:(LOWER_LAYERS + NUM_MID)] = ALPHA_MID_W * theta_star[LOWER_LAYERS:(LOWER_LAYERS + NUM_MID)]  # mid bias
        Q[:, (LOWER_LAYERS + NUM_MID):(LOWER_LAYERS + NUM_MID + NUM_MID*NUM_MID)] = ALPHA_MID_B * theta_star[(LOWER_LAYERS + NUM_MID):(LOWER_LAYERS + NUM_MID + NUM_MID*NUM_MID)]  # mid weight
        Q[:, (LOWER_LAYERS + NUM_MID + NUM_MID*NUM_MID):(LOWER_LAYERS + NUM_MID + NUM_MID*NUM_MID + NUM_CLASSES)] = ALPHA_HIGH_W*theta_star[(LOWER_LAYERS + NUM_MID + NUM_MID*NUM_MID):(LOWER_LAYERS + NUM_MID + NUM_MID*NUM_MID + NUM_CLASSES)] #high bias
        Q[:, (LOWER_LAYERS + NUM_MID + NUM_MID*NUM_MID + NUM_CLASSES):] = ALPHA_HIGH_B*theta_star[(LOWER_LAYERS + NUM_MID + NUM_MID*NUM_MID + NUM_CLASSES):] #high weight
    elif NUM_LAYERS == 1:
        Q[:, LOWER_LAYERS:(LOWER_LAYERS + NUM_CLASSES)] = ALPHA_HIGH_W * theta_star[LOWER_LAYERS:(LOWER_LAYERS + NUM_CLASSES)]  # high bias
        Q[:, (LOWER_LAYERS + NUM_CLASSES):] = ALPHA_HIGH_B * theta_star[(LOWER_LAYERS + NUM_CLASSES):]  # high weight

    flt.compute_inv_sum_diag_dlr(Q)
    flt.update_mPi(theta_star)

    list_of_states.append(state)
    # Save params and Pi_t
    #jnp.save(os.path.join(DIR_RES, 'model_%03i.npy' % (task_id + 1)),
    #         hk.data_structures.to_mutable_dict(state.params))
    #np.savez(os.path.join(DIR_RES, 'Pi_t_%03i.npz' % (task_id + 1)), *flt.Pi_t)
    #print('Saved '+os.path.join(DIR_RES, 'Pi_t_%03i.npz' % (task_id + 1)))

# Save accuracies
np.savetxt(os.path.join(DIR_RES, 'accuracies_matrix.csv'), acc_matrix_laplace_full,
                       delimiter=',', comments='')
np.savetxt(os.path.join(DIR_RES, 'accuracies_avr.csv'), avg_test_accs_laplace_full,
                       delimiter=',', comments='')
