import jax
import os
import time
import argparse
import pandas as pd
import sys
from collections import defaultdict
from typing import Optional
import tensorflow as tf
import wilds
import numpy as np
import jax.numpy as jnp
import jax
from scipy.sparse.linalg import svds
from scipy.linalg import svd
import scipy
import matplotlib.pyplot as plt
try:
    import wandb
except Exception as e:
    pass
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import numpy as np
from types import SimpleNamespace
from flax.training import train_state, checkpoints
from typing import Any, Sequence, Callable, NamedTuple, Optional, Tuple
from flax import linen as nn
import jax

## Standard libraries
import os
import numpy as np
from PIL import Image
from typing import Any
from collections import defaultdict
import time

## Imports for plotting
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0

## Progress bar
from tqdm.auto import tqdm

## To run JAX on TPU in Google Colab, uncomment the two lines below
# import jax.tools.colab_tpu
# jax.tools.colab_tpu.setup_tpu()

## JAX
import jax
import jax.numpy as jnp
from jax import random
# Seeding for random operations

## Flax (NN in JAX)
import flax

from flax import linen as nn
from flax.training import train_state, checkpoints

## Optax (Optimizers in JAX)
import optax

import tensorflow_datasets as tfds

XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
TF_DETERMINISTIC_OPS=1
TF_CUDNN_DETERMINISTIC=1

import yaml
import argparse
import random as rd
parser = argparse.ArgumentParser(description='training')
COMP = '/home/user'
default_config = '/configs/config_default.yaml'
parser.add_argument('--config', type=str, default=COMP+default_config,
                    help='Path config file specifying model '
                         'architecture and training procedure')
args = parser.parse_args()


with open(args.config, 'r') as stream:
    config = yaml.load(stream, yaml.FullLoader)

SEED = config['architecture']['seed']
LAMBDA = config['training']['lam']


Q_c0_kernel = config['training']['c0_kernel']
Q_c0_bias = config['training']['c0_bias']
Q_c1_kernel = config['training']['c1_kernel']
Q_c1_bias = config['training']['c1_bias']
Q_c2_kernel = config['training']['c2_kernel']
Q_c2_bias = config['training']['c2_bias']
Q_d0_kernel = config['training']['d0_kernel']
Q_d0_bias = config['training']['d0_bias']
Q_d1_kernel = config['training']['d1_kernel']
Q_d1_bias = config['training']['d1_bias']
print(Q_c0_kernel, Q_c0_bias, Q_c1_kernel, Q_c1_bias, Q_c2_kernel, Q_c2_bias)
print(Q_d0_kernel, Q_d0_bias, Q_d1_kernel, Q_d1_bias)


print(config['paths'])
DIR_SRC = config['paths']['dir_src']
DIR_EXP = config['paths']['dir_exp']

main_rng = random.PRNGKey(SEED)
rd.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)


HOSPITAL_ID = 0
VAL_CENTER = 0
TEST_CENTER = 1
task_name = 'gradual_lightning'

class CamelyonDataset():
    # _versions_dict = {
    #    '1.0': {
    #        'download_url': 'https://worksheets.codalab.org/rest/bundles/0xe45e15f39fb54e9d9e919556af67aabe/contents/blob/',
    #        'compressed_size': 10_658_709_504}}

    def __init__(self, version='1.0', root_dir='data', download=False, perc=0, which=0, task_name='percentages'):
        self._version = version
        self._dataset_name = 'camelyon17'
        print(os.getcwd())
        self._data_dir = self.initialize_data_dir(root_dir, download)
        self._original_resolution = (96, 96)
        self._perc = perc
        self.which = which
        self.task_name = task_name

        # Read in metadata
        print('self._data_dir', self._data_dir)
        self._metadata_df = pd.read_csv(
            os.path.join(self._data_dir, 'metadata.csv'),
            index_col=0,
            dtype={'patient': 'str'})

        # Get the y values
        self._y_array = self._metadata_df['tumor'].values
        self._y_size = 1
        self._n_classes = 2

        self._split_dict = {
            'train': 2,
            'test': 1,
            'not_used': 0
        }
        self._split_names = {
            'train': 'Train',
            'test': 'Test',
            'not_used': 'NotUsed'
        }
        # Extract splits
        centers = self._metadata_df['center'].values
        num_centers = int(np.max(centers)) + 1

        # Initialize split column
        self._metadata_df['split'] = self._split_dict['not_used']

        # Assign train and test splits for group 0_1_2
        # num_samples = np.sum(mask)
        num_samples = 34904
        if self.task_name == 'percentages':
            center_mask = self._metadata_df['center'] == 0
            self._assign_split(center_mask, num_samples, perc=self._perc, main_group=True)
            center_mask = self._metadata_df['center'] == 1
            self._assign_split(center_mask, num_samples, perc=self._perc, main_group=False)
        elif self.task_name == 'new_labels':
            center_mask = self._metadata_df['center'] < 10
            self._assign_split(center_mask, num_samples, perc=self._perc, main_group=True)
        elif self.task_name == 'corners':
            center_mask = self._metadata_df['center'] == 1
            self._assign_split(center_mask, num_samples, perc=self._perc, main_group=True)
        elif self.task_name == 'gradual_lightning':
            center_mask = self._metadata_df['center'] < 1
            self._assign_split(center_mask, num_samples, perc=self._perc, main_group=True)
        elif self.task_name == 'imbalanced':
            center_mask = self._metadata_df['center'] == 0
            num_samples = sum(center_mask)
            self._assign_split(center_mask, num_samples, perc=self._perc, main_group=True)
            center_mask = self._metadata_df['center'] == 1
            num_samples = sum(center_mask)
            self._assign_split(center_mask, num_samples, perc=self._perc, main_group=False)




        # Update metadata array
        self._split_array = self._metadata_df['split'].values
        self._metadata_array = np.stack((centers, self._metadata_df['slide'].values, self._y_array), axis=1)
        self._metadata_fields = ['hospital', 'slide', 'y']

        filtered_df = self._metadata_df[self._metadata_df['split'] == 2].sample(frac=1)
        file_paths_train = [
            os.path.join(self._data_dir,
                         f'patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png')
            for patient, node, x, y in filtered_df[['patient', 'node', 'x_coord', 'y_coord']].itertuples(index=False,
                                                                                                         name=None)
        ]
        labels_train = list(filtered_df['tumor'].values)
        metadata_list_train = list(filtered_df['center'].values)

        filtered_df = self._metadata_df[self._metadata_df['split'] == 1]
        file_paths_test = [
            os.path.join(self._data_dir,
                         f'patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png')
            for patient, node, x, y in filtered_df[['patient', 'node', 'x_coord', 'y_coord']].itertuples(index=False,
                                                                                                         name=None)
        ]
        labels_test = list(filtered_df['tumor'].values)
        metadata_list_test = list(filtered_df['center'].values)

        dataset_train = tf.data.Dataset.from_tensor_slices((file_paths_train, labels_train, metadata_list_train))
        dataset_test = tf.data.Dataset.from_tensor_slices((file_paths_test, labels_test, metadata_list_test))

        def parse_function(file_path, label, metadata):
            # Read and decode the image file
            image = tf.io.read_file(file_path)
            image = tf.image.decode_png(image, channels=3)  # Adjust channels based on your image format
            image = image / 255
            _DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
            _DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD = [0.229, 0.224, 0.225]


            if self.task_name == 'new_labels':
                label = label + int(5*self.which)

            elif self.task_name == 'corners':
                if self.which == 0:
                    # Crop the image to the top-left corner (10x10)
                    cropped_image = image[:5, :5, :]
                elif self.which == 1:
                    # Crop the image to the top-right corner (10x10)
                    cropped_image = image[:5, -5:, :]
                elif self.which == 2:
                    # Crop the image to the bottom-left corner (10x10)
                    cropped_image = image[-5:, :5, :]
                elif self.which == 3:
                    # Crop the image to the bottom-right corner (10x10)
                    cropped_image = image[-5:, -5:, :]
                else:
                    raise ValueError("Invalid value for which. It should be between 0 and 3.")

                    # Get the shape of the cropped image
                cropped_height = tf.shape(cropped_image)[0]
                cropped_width = tf.shape(cropped_image)[1]

                # Pad the cropped image with zeros to maintain the original shape
                image = tf.pad(cropped_image, paddings=[[0, tf.maximum(0, tf.shape(image)[0] - cropped_height)],
                                                               [0, tf.maximum(0, tf.shape(image)[1] - cropped_width)],
                                                               [0, 0]])

            elif self.task_name == 'gradual_lightning':
                    image = tf.image.adjust_brightness(image, delta=0.5*(-(9-self.which)*0.1 + 0.1*self.which))
                    image = tf.clip_by_value(image, 0.0, 1.0)

            image = (image - _DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN) / _DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD
            return image, label, metadata

        dataset_train = dataset_train.map(parse_function)
        dataset_test = dataset_test.map(parse_function)
        # dataset = dataset.shuffle(buffer_size=len(file_paths))

        # Batch the datasets
        batch_size = 32
        self.dataset_train = dataset_train.batch(batch_size)
        self.dataset_test = dataset_test.batch(batch_size)

    def initialize_data_dir(self, root_dir, download):
        os.makedirs(root_dir, exist_ok=True)
        data_dir = os.path.join(root_dir, f'{self._dataset_name}_v{self._version}')
        version_file = os.path.join(data_dir, f'RELEASE_v{self._version}.txt')


        # If the dataset exists at root_dir, then don't download

        return data_dir

    def _assign_split(self, mask, num_samples, perc, main_group):
        # Determine the number of samples for train and test in the masked subset

        num_train = int(num_samples * 0.9)
        num_test = num_samples - num_train

        # Adjust train/test ratio based on frac and group type
        if main_group:
            num_train_main = int(num_train * (1 - perc))
            num_test_main = int(num_test * (1 - perc))
        else:
            num_train_main = int(num_train * perc)
            num_test_main = int(num_test * perc)

        # Randomly assign samples to train or test
        indices = np.where(mask)[0]
        np.random.shuffle(indices)
        train_indices = indices[:num_train_main]
        test_indices = indices[num_train_main:num_train_main + num_test_main]

        # Update the split values in the dataframe
        self._metadata_df.loc[train_indices, 'split'] = self._split_dict['train']
        self._metadata_df.loc[test_indices, 'split'] = self._split_dict['test']


class TrainerModule:

    def __init__(self,
                 model_name : str,
                 model_class : nn.Module,
                 model_hparams : dict,
                 optimizer_name : str,
                 optimizer_hparams : dict,
                 exmp_imgs : Any,
                 seed):
        """
        Module for summarizing all training functionalities for classification on CIFAR10.

        Inputs:
            model_name - String of the class name, used for logging and saving
            model_class - Class implementing the neural network
            model_hparams - Hyperparameters of the model, used as input to model constructor
            optimizer_name - String of the optimizer name, supporting ['sgd', 'adam', 'adamw']
            optimizer_hparams - Hyperparameters of the optimizer, including learning rate as 'lr'
            exmp_imgs - Example imgs, used as input to initialize the model
            seed - Seed to use in the model initialization
        """
        super().__init__()
        self.model_name = model_name
        self.model_class = model_class
        self.model_hparams = model_hparams
        self.optimizer_name = optimizer_name
        self.optimizer_hparams = optimizer_hparams
        self.seed = seed
        # Create empty model. Note: no parameters yet
        self.model = self.model_class(**self.model_hparams)
        # Prepare logging
        self.log_dir = os.path.join(CHECKPOINT_PATH, self.model_name)
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model(exmp_imgs)

    def create_functions(self):
        # Function to calculate the classification loss and accuracy for a model
        def calculate_loss(params, batch_stats, batch, mPi, Pi_t, train, lam):
            imgs, labels, metadata = batch
            # Run model. During training, we need to update the BatchNorm statistics.
            outs = self.model.apply({'params': params, 'batch_stats': batch_stats},
                                    imgs,
                                    train=train,
                                    mutable=['batch_stats'] if train else False)

            logits, new_model_state = outs if train else (outs, None)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

            def l2_regulariser(params):
                params_flat, _ = jax.flatten_util.ravel_pytree(params)
                diff = params_flat @ params_flat.T
                return 0.5 * diff

            #reg
            def laplace_regulariser(params, mPi: jax.Array, Pi_t: list):
                params_flat = jax.flatten_util.ravel_pytree(params)[0][None, :]
                first = params_flat * Pi_t[0] @ params_flat.T
                second = ((params_flat @ Pi_t[1]) @ Pi_t[2]) @ Pi_t[1].T @ params_flat.T
                third = 2 * params_flat @ mPi
                result = 0.5 * (first + second - third)
                return jnp.sum(result), (first, second, third)

            l2 = l2_regulariser(params)

            reg, others_reg = laplace_regulariser(params, mPi, Pi_t)
            loss_full = loss + + 0*l2 +lam*reg #TODO value of reg
            acc = (logits.argmax(axis=-1) == labels).mean()
            return loss_full, (acc, new_model_state, loss, reg, others_reg, l2)


        # Training function
        def train_step(state, batch, mPi, Pi_t, lam):
            loss_fn = lambda params: calculate_loss(params, state.batch_stats, batch, mPi, Pi_t, train=True,lam=lam)

            # Get loss, gradients for loss, and other outputs of loss function
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            loss, acc, new_model_state, loss_part, reg, others_reg, l2 = ret[0], *ret[1]
            # Update parameters and batch statistics
            state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
            return state, loss, acc, (loss_part, reg, others_reg, l2)
        # Eval function
        def eval_step(state, batch, mPi, Pi_t,lam):
            # Return the accuracy for a single batch
            _, (acc, _, _,_,_,_) = calculate_loss(state.params, state.batch_stats, batch,mPi, Pi_t, train=False, lam=lam)
            return acc
        # jit for efficiency
        #print('here', jax.devices('cpu')[0], jax.devices('gpu')[0])
        self.train_step = jax.jit(train_step, device=jax.devices('gpu')[0])
        self.eval_step = jax.jit(eval_step, device=jax.devices('gpu')[0])

    def calculate_loss_alone(self, logits, batch, train):
        imgs, labels, metadata = batch

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        return loss


    def init_model(self, exmp_imgs):
        # Initialize model
        init_rng = jax.random.PRNGKey(self.seed)
        variables = self.model.init(init_rng, exmp_imgs, train=True)
        self.init_params, self.init_batch_stats = variables['params'], variables['batch_stats']
        self.state = None

    def init_optimizer(self, num_epochs, num_steps_per_epoch):
        # Initialize learning rate schedule and optimizer
        if self.optimizer_name.lower() == 'adam':
            opt_class = optax.adam
        elif self.optimizer_name.lower() == 'adamw':
            opt_class = optax.adamw
        elif self.optimizer_name.lower() == 'sgd':
            opt_class = optax.sgd
        else:
            assert False, f'Unknown optimizer "{self.optimizer_name}"'
        # We decrease the learning rate by a factor of 0.1 after 60% and 85% of the training
        lr_schedule = optax.piecewise_constant_schedule(
            init_value=self.optimizer_hparams.pop('lr'),
            boundaries_and_scales=
                {int(num_steps_per_epoch*num_epochs*0.6): 0.1,
                 int(num_steps_per_epoch*num_epochs*0.85): 0.1}
        )
        # Clip gradients at max value, and evt. apply weight decay
        transf = [optax.clip(1.0)]
        if opt_class == optax.sgd and 'weight_decay' in self.optimizer_hparams:  # wd is integrated in adamw
            transf.append(optax.add_decayed_weights(self.optimizer_hparams.pop('weight_decay')))
        optimizer = optax.chain(
            *transf,
            opt_class(lr_schedule, **self.optimizer_hparams)
        )
        # Initialize training state
        self.state = TrainState.create(apply_fn=self.model.apply,
                                       params=self.init_params if self.state is None else self.state.params,
                                       batch_stats=self.init_batch_stats if self.state is None else self.state.batch_stats,
                                       tx=optimizer)

    def train_model(self, train_loader, test_loader, ti, num_epochs=200, flt=None, lam=0):
        # Train model for defined number of epochs
        # We first need to create optimizer and the scheduler for the given number of epochs
        if ti == 0:
            self.init_optimizer(num_epochs, len(train_loader))
            flat_params_init, _ = jax.flatten_util.ravel_pytree(self.state.params)
            flt = Diag_LowRank(flat_params_init.size, 2) #k*num_classes
        # Track best eval accuracy
        best_eval = 0.0
        print('lam',lam)
        for epoch_idx in tqdm(range(1, num_epochs+1), position=0, leave=True):
            self.train_epoch(train_loader, ti, epoch=epoch_idx, flt=flt, lam=lam)
            if epoch_idx % 2 == 0:
                eval_acc = self.eval_model(test_loader, flt, lam)
                wandb.log({"test/acc_task"+str(ti): eval_acc})
                if eval_acc >= best_eval:
                    best_eval = eval_acc
                    #self.save_model(step=epoch_idx)
        return flt

    def train_epoch(self, train_loader, ti, epoch, flt, lam):
        # Train model for one epoch, and log avg loss and accuracy
        metrics = defaultdict(list)
        for batch in tqdm(train_loader, desc='Training', position=0, leave=True):
            images, labels, metadata = batch
            self.state, loss, acc, (loss_part, reg, others_reg, l2) = self.train_step(self.state, batch, flt.mPi, flt.Pi_t, lam=lam)
            metrics['loss'].append(loss)
            metrics['acc'].append(acc)
            metrics['reg'].append(reg)
            metrics['loss_no_reg'].append(loss_part)
            metrics['l2'].append(l2)


        for key in metrics:
            avg_val = np.stack(jax.device_get(metrics[key])).mean()
            wandb.log({"train/"+key+"_task"+str(ti): avg_val})

    def eval_model(self, data_loader, flt=None, lam=0):
        # Test model on all images of a data loader and return avg loss
        correct_class, count = 0, 0
        for batch in data_loader:
            images, labels, metadata = batch
            acc = self.eval_step(self.state, batch, flt.mPi, flt.Pi_t, lam)
            correct_class += acc * batch[0].shape[0]
            count += batch[0].shape[0]
        eval_acc = (correct_class / count).item()
        return eval_acc

    def save_model(self, step=0):
        # Save current model at certain training iteration
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir,
                                    target={'params': self.state.params,
                                            'batch_stats': self.state.batch_stats},
                                    step=step,
                                   overwrite=True)

    def load_model(self, pretrained=False):
        # Load model. We use different checkpoint for pretrained models
        if not pretrained:
            state_dict = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=None)
        else:
            state_dict = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(CHECKPOINT_PATH, f'{self.model_name}.ckpt'), target=None)
        self.state = TrainState.create(apply_fn=self.model.apply,
                                       params=state_dict['params'],
                                       batch_stats=state_dict['batch_stats'],
                                       tx=self.state.tx if self.state else optax.sgd(0.1)   # Default optimizer
                                      )

    def checkpoint_exists(self):
        # Check whether a pretrained model exist for this autoencoder
        return os.path.isfile(os.path.join(CHECKPOINT_PATH, f'{self.model_name}.ckpt'))




    def compute_GGN(self, batch, train=False):
        def J_wrapper_function(flattened_weights, func_to_unflatten, image, train):
            unflattened_weights = func_to_unflatten(flattened_weights)

            outs = self.model.apply({'params': unflattened_weights, 'batch_stats': self.state.batch_stats},
                                    image,
                                    train=train,
                                    mutable=['batch_stats'] if train else False)
            logits, new_model_state = outs if train else (outs, None)
            return logits

        images, labels, metadata = batch
        flat_params, tree_str = jax.flatten_util.ravel_pytree(self.state.params)
        J = jax.jacobian(J_wrapper_function)(flat_params, tree_str, images, train)

        outs = self.model.apply({'params': self.state.params, 'batch_stats': self.state.batch_stats},
                                    images,
                                    train=train,
                                    mutable=['batch_stats'] if train else False)
        logits, new_model_state = outs if train else (outs, None)
        H = jax.hessian(self.calculate_loss_alone)(logits, (images, labels, metadata ), train)
        H = H[np.arange(0,H.shape[0]),:,np.arange(0,H.shape[0]),:]
        H = np.array(H)
        #H[H < 0] = 0

        return J, H




def train_classifier(*args, ti=0, trainer_p=None, num_epochs=200, seed=0, flt=None, lam_t=0, **kwargs):
    # Create a trainer module with specified hyperparameters
    if ti == 0:
        trainer = TrainerModule(*args,seed=seed, **kwargs)
    else:
        trainer = trainer_p
    if not trainer.checkpoint_exists():  # Skip training if pretrained model exists
        flt = trainer.train_model(train_loader, test_loader, ti, num_epochs=num_epochs, flt=flt, lam=lam_t)
        #trainer.load_model()
    else:
        pass
        #trainer.load_model(pretrained=True)
    # Test trained model
    test_acc = trainer.eval_model(test_loader, flt, lam_t)
    batch = next(iter(train_loader))
    J, H = trainer.compute_GGN(batch, False)

    return trainer, {'test': test_acc}, (J,H), flt

densenet_kernel_init = nn.initializers.kaiming_normal()

class DenseLayer(nn.Module):
    bn_size : int  # Bottleneck size (factor of growth rate) for the output of the 1x1 convolution
    growth_rate : int  # Number of output channels of the 3x3 convolution
    act_fn : callable  # Activation function

    @nn.compact
    def __call__(self, x, train=True):

        z = nn.BatchNorm()(x, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(self.bn_size * self.growth_rate,
                    kernel_size=(1, 1),
                    kernel_init=densenet_kernel_init,
                    use_bias=False)(z)
        z = nn.BatchNorm()(z, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(self.growth_rate,
                    kernel_size=(3, 3),
                    kernel_init=densenet_kernel_init,
                    use_bias=False)(z)
        x_out = jnp.concatenate([x, z], axis=-1)

        return x_out

class DenseBlock(nn.Module):
    num_layers : int  # Number of dense layers to apply in the block
    bn_size : int  # Bottleneck size to use in the dense layers
    growth_rate : int  # Growth rate to use in the dense layers
    act_fn : callable  # Activation function to use in the dense layers

    @nn.compact
    def __call__(self, x, train=True):
        for _ in range(self.num_layers):
            x = DenseLayer(bn_size=self.bn_size,
                           growth_rate=self.growth_rate,
                           act_fn=self.act_fn)(x, train=train)
        return x

class TransitionLayer(nn.Module):
    c_out : int  # Output feature size
    act_fn : callable  # Activation function

    @nn.compact
    def __call__(self, x, train=True):
        x = nn.BatchNorm()(x, use_running_average=not train)
        x = self.act_fn(x)
        x = nn.Conv(self.c_out,
                    kernel_size=(1, 1),
                    kernel_init=densenet_kernel_init,
                    use_bias=False)(x)
        x = nn.avg_pool(x, (2, 2), strides=(2, 2))
        return x

class DenseNet(nn.Module):
    num_classes : int
    act_fn : callable = nn.relu
    num_layers : tuple = (6, 6, 6, 6)
    bn_size : int = 2
    growth_rate : int = 16

    @nn.compact
    def __call__(self, x, train=True):
        c_hidden = self.growth_rate * 2  # The start number of hidden channels
        x = nn.Conv(c_hidden,
                    kernel_size=(7, 7),
                    strides=(2,2),
                    kernel_init=densenet_kernel_init)(x)
        x = nn.BatchNorm()(x, use_running_average=not train)
        x = self.act_fn(x)

        x = nn.max_pool(x, (3, 3), strides=(2, 2))

        for block_idx, num_layers in enumerate(self.num_layers):
            x = DenseBlock(num_layers=num_layers,
                           bn_size=self.bn_size,
                           growth_rate=self.growth_rate,
                           act_fn=self.act_fn)(x, train=train)
            c_hidden += num_layers * self.growth_rate
            if block_idx < len(self.num_layers)-1:  # Don't apply transition layer on last block
                x = TransitionLayer(c_out=c_hidden//2,
                                    act_fn=self.act_fn)(x, train=train)
                c_hidden //= 2

        x = nn.BatchNorm()(x, use_running_average=not train)
        x = self.act_fn(x)
        x = x.mean(axis=(1, 2))
        x = nn.Dense(self.num_classes)(x)
        return x

class CNN(nn.Module):
    num_classes : int
    @nn.compact
    def __call__(self, x, train=True):

        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.BatchNorm()(x, use_running_average=not train)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))

        x = nn.Conv(features=16, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.BatchNorm()(x, use_running_average=not train)
        x = nn.max_pool(x, window_shape=(4, 4), strides=(4, 4))

        x = nn.Conv(features=4, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.BatchNorm()(x, use_running_average=not train)
        x = nn.max_pool(x, window_shape=(4, 4), strides=(4, 4))

        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(features=8)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x

class TrainState(train_state.TrainState):
    # A simple extension of TrainState to also include batch statistics
    batch_stats: Any





class Diag_LowRank(object):

    def __init__(self, shape_weights,num_classes):
        print(num_classes)
        self.mPi = jnp.zeros((shape_weights,1))
        self.Pi_t = [jnp.ones((1, shape_weights)), jnp.zeros((shape_weights,num_classes)), jnp.zeros((num_classes,num_classes))]

    def add_diag(self, D):
        self.Pi_t[0] += D
        # update the SVD of the whole thing

    def block_matrix(self, C, H):
        b, c, _ = C.shape

        # Create block matrices using broadcasting
        upper_block = jnp.concatenate([C, jnp.zeros_like(C)], axis=2)
        lower_block = jnp.concatenate([jnp.zeros_like(H), H], axis=2)

        return jnp.concatenate([upper_block, lower_block], axis=1)

    def add_low(self, J, H):
        U = self.Pi_t[1]
        C = self.Pi_t[2]
        left_vec = np.zeros((J.shape[2], J.shape[0]+1, C.shape[0])) # d, b, c
        C_12 = scipy.linalg.sqrtm(C)
        left_vec[:,0,:] = U @ C_12
        for b in range(J.shape[0]):
            H_12_b = scipy.linalg.sqrtm(H[b]+1e-4*jnp.eye(H.shape[1]))
            left_vec[:,b+1,:J.shape[1]] = (1/np.sqrt(J.shape[0]))*J[b].T @ H_12_b
        new_Ul, new_Sl, new_Vl = self.truncated_svd(left_vec, m=C.shape[0])


        self.Pi_t[1], self.Pi_t[2] = new_Ul, jnp.diag(new_Sl**2)


    def truncated_svd(self, M, m):
        M_reshaped = M.reshape(M.shape[0], M.shape[1] * M.shape[2])
        U, s, Vt = svds(np.array(M_reshaped) , k=m)
        #U_exact, s_exac, Vt_exact = svd(np.array(M_reshaped), full_matrices=False)
        idx = np.argsort(s)[::-1]
        U = U[:, idx]
        s = s[idx]
        Vt = Vt[idx, :]
        return U, s, Vt

    def compute_inv_diag(self, D):
        return jnp.diag(1.0 / jnp.diag(D))

    def compute_inv_sum_diag_dlr(self, Q):
        A_1 = (1/(self.Pi_t[0]))
        L = (1/(Q + A_1))
        Up = A_1.T * self.Pi_t[1]
        Cp = jnp.diag(1/jnp.diag(self.Pi_t[2])) + (self.Pi_t[1].T @ Up)
        left = L.T * Up
        mid = (Cp - Up.T @ left)
        mid_inv = jnp.linalg.pinv(mid)
        self.Pi_t = [L, left, mid_inv]


    def update_mPi(self, theta_star):
        self.mPi = theta_star[None,:] * self.Pi_t[0]
        temp1 = theta_star[None, :] @ self.Pi_t[1]
        temp2 = temp1 @ self.Pi_t[2]
        self.mPi += (temp2 @ self.Pi_t[1].T)
        self.mPi = self.mPi.T




CHECKPOINT_PATH = '/home/user/runs'

wandb.init(
    # set the wandb project where this run will be logged
    project="exp",

    # track hyperparameters and run metadata
    config={'seed':SEED, 'lam':LAMBDA
    }
)

root_dir = 'data'

def zeros_like(x):
    return jnp.zeros_like(x)

DIR_RES = DIR_SRC+DIR_EXP
print(DIR_RES)
percentages = [0,0,0,0]
T= 8
list_of_test_loaders = []

acc_matrix_laplace_full = np.zeros((T, T))
avg_test_accs_laplace_full = []
download_datasets = 1

for ti in range(T):

    full_dataset = CamelyonDataset(root_dir=root_dir, perc=percentages[0], which=ti, task_name=task_name)
    train_loader = tfds.as_numpy(full_dataset.dataset_train)
    test_loader = tfds.as_numpy(full_dataset.dataset_test).append(test_loader)
    if ti == 0:
        lam_t = 0
    else:
        lam_t = LAMBDA
        print(lam_t)

    images, labels, metadata = next(iter(train_loader))

    if ti == 0:
        cnn_trainer, cnn_results, (J, H), flt = train_classifier(model_name="CNN",
                                                    model_class=CNN,
                                                    ti = ti,
                                                    trainer_p = None,
                                                    model_hparams={"num_classes": 2},
                                                    optimizer_name="adam",
                                                    optimizer_hparams={"lr": 1e-3},
                                                    exmp_imgs=jax.device_put(images),
                                                    num_epochs=5,
                                                    seed = SEED,
                                                    flt=None,
                                                    lam_t = lam_t)
    else:
        cnn_trainer, cnn_results, (J,H), flt = train_classifier(model_name="CNN",
                                                    model_class=CNN,
                                                    ti=ti,
                                                    trainer_p=cnn_trainer,
                                                    model_hparams = {"num_classes": 2},
                                                    optimizer_name = "adam",
                                                    optimizer_hparams = {"lr": 1e-3},
                                                    exmp_imgs = jax.device_put(images),
                                                    num_epochs =5,
                                                    seed=SEED,
                                                    flt=flt,
                                                    lam_t = lam_t)



    for tii in range(ti+1):
        acc = cnn_trainer.eval_model(list_of_test_loaders[tii], flt, lam_t)
        wandb.log({"eval/acc_task"+str(ti): acc})
        acc_matrix_laplace_full[ti, tii] = acc
        print('Current task:', ti, 'Evaluated on:', tii)
        print(f'Test Accuracy (%): {acc_matrix_laplace_full[ti, tii]:.2f}).')
    avg_test_acc_t_laplace_full = acc_matrix_laplace_full[tii, :(tii + 1)].mean()
    avg_test_accs_laplace_full.append(avg_test_acc_t_laplace_full)

    #new part with reg
    theta_star, _ = jax.flatten_util.ravel_pytree(cnn_trainer.state.params)
    if ti ==0:
        theta_star_old = jax.tree_util.tree_map(zeros_like, cnn_trainer.state.params)
    #plot_params(cnn_trainer.state.params)

    theta_star_old = cnn_trainer.state.params
    # Update
    flt.add_low(J, H)
    # Predict
    Q = np.zeros((1, theta_star.size))
    bn = 32+32+16+16+4+4
    c0_kernel = bn + 864
    c0_bias = c0_kernel + 32
    c1_kernel = c0_bias + 4608
    c1_bias = c1_kernel + 16
    c2_kernel = c1_bias + 576
    c2_bias = c2_kernel + 4
    d0_kernel = c2_bias + 128
    d0_bias = d0_kernel + 8
    d1_kernel = d0_bias + 16
    cv0 = bn + (864+32)
    cv = bn + (864 + 32 + 4608 + 16 + 576 + 4)
    dense = bn + cv # 128+8+16+2

    Q[:bn] = 0
    Q[bn:c0_kernel] = Q_c0_kernel * (theta_star[bn:c0_kernel] ** 2).mean()
    Q[c0_kernel:c0_bias] = Q_c0_bias * (theta_star[c0_kernel:c0_bias] ** 2).mean()
    Q[c1_bias:c1_kernel] = Q_c1_kernel * (theta_star[c1_bias:c1_kernel] ** 2).mean()
    Q[c1_kernel:c1_bias] = Q_c1_bias * (theta_star[c1_kernel:c1_bias] ** 2).mean()
    Q[c1_bias:c2_kernel] = Q_c2_kernel * (theta_star[c1_bias:c2_kernel] ** 2).mean()
    Q[c2_kernel:c2_bias] = Q_c2_bias * (theta_star[c2_kernel:c2_bias] ** 2).mean()
    Q[c2_bias:d0_kernel] = Q_d0_kernel * (theta_star[c2_bias:d0_kernel] ** 2).mean()
    Q[d0_kernel:d0_bias] = Q_d0_bias * (theta_star[d0_kernel:d0_bias] ** 2).mean()
    Q[d0_bias:d1_kernel] = Q_d1_kernel * (theta_star[d0_bias:d1_kernel] ** 2).mean()
    Q[d1_kernel:] = Q_d1_bias * (theta_star[d1_kernel:] ** 2).mean()

    flt.compute_inv_sum_diag_dlr(Q)
    flt.update_mPi(theta_star)


if not os.path.exists(DIR_RES):
    os.makedirs(DIR_RES)

np.savetxt(os.path.join(DIR_RES, 'accuracies_matrix.csv'), acc_matrix_laplace_full,
                       delimiter=',', comments='')
np.savetxt(os.path.join(DIR_RES, 'accuracies_avr.csv'), avg_test_accs_laplace_full,
                       delimiter=',', comments='')

