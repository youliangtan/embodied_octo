#!/usr/bin/env python3

import jax
import jax.numpy as jnp
from jax import tree_util
from jax import random, grad
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from flax import linen as nn
from flax.training import train_state
import optax
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from transformer import TransformerPredictor
from dataset import TrajectoryDataset
import os
import tensorflow as tf

# export XLA_PYTHON_CLIENT_PREALLOCATE="false"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"


print_yellow = lambda x: print("\033[33m", x, "\033[0m")

class ConvEncoder(nn.Module):
    latent_dim: int = 16

    @nn.compact
    def __call__(self, x):
        # 2 conv, max_and 2 dense
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        # flatten
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.latent_dim)(x)
        return x


class Conv3DEncoder(nn.Module):
    latent_dim: int = 64

    @nn.compact
    def __call__(self, x):
        # 2 conv, max_and 2 dense
        x = nn.Conv(features=16, kernel_size=(3, 3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3, 3))(x)
        x = nn.max_pool(x, window_shape=(2, 2, 2), strides=(2, 2, 2))
        # flatten
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.latent_dim)(x)
        return x


class TrajectoryEncoder(nn.Module):
    """
    trajectory encoder uses the Conv3DEncoder,
    and stack an action array of size 7, then MLP
    """
    img_latent_dim: int = 64
    encoding_dim: int = 16

    @nn.compact
    def __call__(self, image, action):
        # 3D Convolutional Encoder
        x = Conv3DEncoder(latent_dim=self.img_latent_dim)(image)
        # flatten the action
        action = action.reshape((action.shape[0], -1))
        x = jnp.hstack([x, action])  # now you can concatenate x and action
        # MLP
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.encoding_dim)(x)
        return x


class CalibrationEncoder(nn.Module):
    """
    calibration encoder is a simple MLP
    that takes in the calibration data and outputs the encoding
    """
    encoding_dim: int = 16

    @nn.compact
    def __call__(self, calibration):
        x = nn.Dense(16)(calibration)
        x = nn.relu(x)
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        x = nn.Dense(self.encoding_dim)(x)
        return x


def count_params(params):
    """Count the total number of parameters."""
    params_flat, _ = tree_util.tree_flatten(params)
    return sum(p.size for p in params_flat)


def normalize(x, min_val, max_val):
    """Normalize data to [0, 1] range."""
    return (x - min_val) / (max_val - min_val)


def denormalize(x, min_val, max_val):
    """Denormalize data from [0, 1] range."""
    return x * (max_val - min_val) + min_val


def validate_model(test_ds_iter, model, state, norm_min, norm_max, num_samples=10):
    diff = jnp.zeros((7))
    mse = 0
    for i in range(num_samples):
        data = next(test_ds_iter)
        images = jnp.array(data['images'])
        states = jnp.array(data['states'])
        cam_profile = jnp.array(data['cam_profile'])

        norm_cam_profile = normalize(cam_profile, norm_min, norm_max)
        preds = model.apply({'params': state.params}, images, states)
        preds_denorm = denormalize(preds, norm_min, norm_max)
        diff += jnp.mean(jnp.abs(preds_denorm - cam_profile), axis=0)
        mse += jnp.mean(jnp.square(preds - norm_cam_profile))

    diff /= num_samples
    print(f" - Val | Average Diff: {diff}, with avg mse: {mse/num_samples}")


########################################################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Dataset')
    parser.add_argument('--data_path', '-d', type=str,
                        help='Path to training dataset')
    parser.add_argument('--batch_size', '-b', type=int,
                        default=4, help='Batch size')
    parser.add_argument('--num_epochs', '-e', type=int,
                        default=10, help='Number of epochs')
    parser.add_argument('--traj_length', '-t', type=int,
                        default=15, help='Length of trajectory')
    args = parser.parse_args()

    train_dataset_dir = os.path.join(args.data_path, 'train')
    test_dataset_dir = os.path.join(args.data_path, 'val')

    td = TrajectoryDataset(train_dataset_dir,
                           traj_size=args.traj_length)

    devices = jax.local_devices()
    shard_count = len(devices)    
    sharding = jax.sharding.PositionalSharding(devices)
    print(f"Devices: {devices}")

    # create a 1D mesh with a single axis named "batch"
    mesh = Mesh(jax.devices(), axis_names="batch")
    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    data_parallel_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    
    # TODO: complete impl of sharding

    dataset = td.get_dataset()
    # repeat the dataset indefinitely
    dataset = dataset.repeat()
    batched_dataset = dataset.batch(args.batch_size)

    # check the shape of the data
    datapoint = next(iter(batched_dataset))
    images_shape = datapoint['images'].shape
    states_shape = datapoint['states'].shape
    print(f"Images shape: {images_shape}")
    print(f"States shape: {states_shape}")
    assert len(
        images_shape) == 5, "Images shape should be (batch, time, height, width, channels)"
    assert len(
        states_shape) == 3, "States shape should be (batch, time, state_dim)"

    # use iterator to get the data
    iterator = iter(batched_dataset)

    model = TrajectoryEncoder(img_latent_dim=64, encoding_dim=7)
    optimizer = optax.adam(learning_rate=1e-3)

    # Sample first 100 data points to get the normalization range
    data = next(iterator)
    norm_min = data['cam_profile']
    norm_max = data['cam_profile']
    for i in range(100):
        data = next(iterator)
        norm_min = tf.minimum(norm_min, tf.reduce_min(
            data['cam_profile'], axis=0))
        norm_max = tf.maximum(norm_max, tf.reduce_max(
            data['cam_profile'], axis=0))
    norm_max = jnp.array(norm_max)
    norm_min = jnp.array(norm_min)
    print(f"Normalization Min: {norm_min}")
    print(f"Normalization Max: {norm_max}")

    @jax.jit
    def train_step(state, batch):
        def loss_fn(params, l2_reg=1e-8):
            # forward pass,
            image, state, cam_profile = batch['images'], batch['states'], batch['cam_profile']

            # TODO: now assume states as actions, need to change
            preds = model.apply({'params': params}, image, state)

            # L2 regularization
            # TODO: dont use for loop
            l2_reg_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))

            # the loss will be L2 loss of the predicted and the true cam profile
            norm_cam_profile = normalize(cam_profile, norm_min, norm_max)
            mse_loss = jnp.mean(jnp.square(preds - norm_cam_profile))

            # total loss
            loss = mse_loss + l2_reg * l2_reg_loss
            return loss

        # compute the gradient
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(state.params)
        # update the optimizer TODO: is this needed?
        updates, new_opt_state = optimizer.update(grad, state.opt_state)
        # update the model
        new_state = state.apply_gradients(grads=grad)
        return new_state, new_opt_state, loss

    # initialize the model
    rng = random.PRNGKey(0)
    rng, key = random.split(rng)

    init_img_data = jnp.ones(
        (args.batch_size, images_shape[1], images_shape[2], images_shape[3], images_shape[4]))
    init_act_data = jnp.ones(
        (args.batch_size, states_shape[1], states_shape[2]))
    params = model.init(key, init_img_data, init_act_data)['params']

    print(
        f"Total number of parameters: {count_params(params)}, equivalent to {count_params(params) * 4 / 1024 / 1024} MB")

    # opt_state = optimizer.init(params)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    test_dataset = TrajectoryDataset(test_dataset_dir, traj_size=args.traj_length)
    test_dataset = test_dataset.get_dataset()
    test_dataset = test_dataset.batch(args.batch_size).repeat()
    test_ds_iter = iter(test_dataset)

    # training loop
    for epoch in range(args.num_epochs):
        # print(f"----------Epoch {epoch}------------")
        avg_loss = 0
        for i in range(50):  # 50 batches
            data = next(iterator)
            # Convert TensorFlow tensors to NumPy arrays
            data = {k: v.numpy() for k, v in data.items()}
            state, opt_state, loss = train_step(state, data)
            avg_loss += loss
        avg_loss /= 50
        print_yellow(f"Epoch {epoch} | Average Loss: {avg_loss}")
        validate_model(test_ds_iter, model, state, norm_min, norm_max, num_samples=4)

    # save the model
    print("Done training")

    print("\n ---------- Validating Model ----------")
    validate_model(test_ds_iter, model, state, norm_min, norm_max, num_samples=4)
    print("Done testing \n\n")
