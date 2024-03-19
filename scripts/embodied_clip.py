#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import jax.numpy as np
from jax import random, grad
from flax import linen as nn
from flax.training import train_state
import optax
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from transformer import TransformerPredictor
from dataset import TrajectoryDataset


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
        x = nn.Conv(features=32, kernel_size=(3, 3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3, 3))(x)
        x = nn.max_pool(x, window_shape=(2, 2, 2), strides=(2, 2, 2))
        # flatten
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Dataset')
    parser.add_argument('--data_path', '-d', type=str, help='Path to dataset')
    parser.add_argument('--batch_size', '-b', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', '-e', type=int, default=10, help='Number of epochs')
    parser.add_argument('--trajectory_length', '-t', type=int, default=15, help='Length of trajectory')
    args = parser.parse_args()

    td = TrajectoryDataset(args.data_path,
                           traj_size=args.trajectory_length)

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
    assert len(images_shape) == 5, "Images shape should be (batch, time, height, width, channels)"
    assert len(states_shape) == 3, "States shape should be (batch, time, state_dim)"

    # use iterator to get the data
    iterator = iter(batched_dataset)

    model = TrajectoryEncoder(img_latent_dim=64, encoding_dim=7)
    optimizer = optax.adam(learning_rate=1e-3)

    @jax.jit
    def train_step(state, batch):
        def loss_fn(params):
            # forward pass,
            image, state, cam_profile = batch['images'], batch['states'], batch['cam_profile']

            # TODO: now assume states as actions, need to change
            preds = model.apply({'params': params}, image, state)

            # the loss will be L2 loss of the predicted and the true cam profile
            # TODO normalization the predicted cam profile
            loss = jnp.mean(jnp.square(preds - cam_profile))
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
    
    init_img_data = jnp.ones((args.batch_size, images_shape[1], images_shape[2], images_shape[3], images_shape[4]))
    init_act_data = jnp.ones((args.batch_size, states_shape[1], states_shape[2]))
    params = model.init(key, init_img_data, init_act_data)['params']

    # opt_state = optimizer.init(params)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    # training loop
    for epoch in range(args.num_epochs):
        print(f"----------Epoch {epoch}------------")
        avg_loss = 0
        for i in range(50): # 50 batches
            data = next(iterator)
            # Convert TensorFlow tensors to NumPy arrays
            data = {k: v.numpy() for k, v in data.items()}
            state, opt_state, loss = train_step(state, data)
            avg_loss += loss
        avg_loss /= 50
        print(f"Epoch {epoch} | Average Loss: {avg_loss}")

    # save the model
    print("Done training")
