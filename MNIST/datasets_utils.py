from typing import Iterator
from train_utils import *
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import random


def load_dataset(split: str, *,
                 shuffle: bool,
                 batch_size: int,
                 input_dim: int,
                 seed: int,
                 task: int) -> Iterator[Batch]:
    """Loads the MNIST dataset and permute."""
    np.random.seed(task + seed)
    perm_inds = np.arange(input_dim*input_dim)

    # First task is (unpermuted) MNIST, subsequent tasks are random permutations of pixels
    if task > 0:
        np.random.shuffle(perm_inds)

    def tf_permute(x):
        im = x["image"]
        im = tf.reshape(im, [-1])
        im = tf.gather(im, perm_inds)
        im = tf.reshape(im, (input_dim, input_dim, 1))
        x["image"] = im
        return x

    ds, ds_info = tfds.load("mnist:3.*.*", split=split, with_info=True)
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(ds_info.splits[split].num_examples, seed=0)
    ds = ds.repeat()
    ds = ds.map(lambda x: tf_permute(x))  # permute the pixels
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x: Batch(**x))  # turn into NamedTuple
    return iter(tfds.as_numpy(ds))


def load_dataset_split(split: str, *, shuffle: bool, batch_size: int, seed: int,
                                  task: int) -> Iterator[Batch]:
    """Loads the MNIST dataset - downsample and split."""

    random.seed(task + seed)
    chosen_idx = random.sample(range(0, 10), 2)
    chosen_idx = tf.cast(chosen_idx, tf.int64)

    def tf_split(x):
        im = x["image"]
        lab = x["label"]

        placeholder_image = tf.zeros_like(im)
        placeholder_label = tf.constant(-1, dtype=tf.int64)

        condition = tf.reduce_any(tf.equal(chosen_idx, lab))

        def true_fn():
            return {"image": im, "label": lab}

        def false_fn():
            return {"image": placeholder_image, "label": placeholder_label}

        return tf.cond(condition, true_fn, false_fn)

    ds, ds_info = tfds.load("mnist:3.*.*", split=split, with_info=True)
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(ds_info.splits[split].num_examples, seed=0)
    ds = ds.repeat()
    ds = ds.map(tf_split).filter(lambda x: tf.not_equal(x["label"], -1))
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x: Batch(**x))

    return iter(tfds.as_numpy(ds))

def load_dataset_disjoint(split: str, *, shuffle: bool, batch_size: int,
                                  task: int) -> Iterator[Batch]:
    """Loads the MNIST dataset - downsample and disjoint."""

    if task == 0:
        chosen_idx = jnp.array([0,1,2,3,4])
    else: chosen_idx = jnp.array([5,6,7,8,9])
    chosen_idx = tf.cast(chosen_idx, tf.int64)

    def tf_split(x):
        im = x["image"]
        lab = x["label"]

        placeholder_image = tf.zeros_like(im)
        placeholder_label = tf.constant(-1, dtype=tf.int64)

        condition = tf.reduce_any(tf.equal(chosen_idx, lab))

        def true_fn():
            return {"image": im, "label": lab}

        def false_fn():
            return {"image": placeholder_image, "label": placeholder_label}

        return tf.cond(condition, true_fn, false_fn)

    ds, ds_info = tfds.load("mnist:3.*.*", split=split, with_info=True)
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(ds_info.splits[split].num_examples, seed=0)
    ds = ds.repeat()
    ds = ds.map(tf_split).filter(lambda x: tf.not_equal(x["label"], -1))
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x: Batch(**x))

    return iter(tfds.as_numpy(ds))




