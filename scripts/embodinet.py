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

# Load MNIST dataset
# Load and preprocess MNIST dataset

def load_mnist(split: str):
    ds, ds_info = tfds.load('mnist', split=split,
                            as_supervised=True, with_info=True, batch_size=-1)
    np_ds = tfds.as_numpy(ds)
    images, labels = np_ds
    images = images.astype(jnp.float32) / 255.0
    print(images.shape, labels.shape)
    return images, labels


def organize_by_class(images, labels):
    """Organize images by class buckets"""
    return {i: images[labels == i] for i in range(10)}


def get_triplet(class_buckets, rng, batch_size):
    """function to sample a triplet"""
    anchor_imgs, positive_imgs, negative_imgs = [], [], []
    for _ in range(batch_size):
        # generate random numbers
        rng, anchor_rng, positive_rng, negative_rng = random.split(rng, 4)

        # Randomly select a class for anchor and positive
        anchor_class = random.choice(anchor_rng, np.array(range(10))).item()

        # Choose different class for negative
        negative_classes = np.array(
            [i for i in range(10) if i != anchor_class])
        negative_class = random.choice(negative_rng, negative_classes).item()

        # Randomly choose one sample from the selected anchor and positive class
        anchor_idx = random.choice(anchor_rng, len(
            class_buckets[anchor_class]), shape=()).item()
        positive_idx = random.choice(positive_rng, len(
            class_buckets[anchor_class]), shape=()).item()

        # Randomly choose one sample from the selected negative class
        negative_idx = random.choice(negative_rng, len(
            class_buckets[negative_class]), shape=()).item()

        anchor_imgs.append(class_buckets[anchor_class][anchor_idx])
        positive_imgs.append(class_buckets[anchor_class][positive_idx])
        negative_imgs.append(class_buckets[negative_class][negative_idx])

    # Convert lists to jnp arrays
    return jnp.stack(anchor_imgs), jnp.stack(positive_imgs), jnp.stack(negative_imgs)

##############################################################################

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


def make_embodinet(type: str, latent_dim: int):
    if type == "conv":
        return ConvEncoder(latent_dim=latent_dim)
    elif type == "transformer":
        return TransformerPredictor(num_layers=3,
                                    model_dim=64,
                                    num_classes=latent_dim,
                                    num_heads=4,
                                    dropout_prob=0.15,
                                    input_dropout_prob=0.05)
    else:
        raise ValueError("Invalid model type")

# TODO: use different loss function


def triplet_loss(anchor, positive, negative, margin=0.2):
    """Define triplet loss"""
    pos_dist = jnp.sum(jnp.square(anchor - positive), axis=1)
    neg_dist = jnp.sum(jnp.square(anchor - negative), axis=1)
    basic_loss = pos_dist - neg_dist + margin
    loss = jnp.maximum(basic_loss, 0.0)
    return jnp.mean(loss)

##############################################################################


MODEL_TYPE = "conv"  # TODO: configurable
MODEL_TYPE = "transformer"
BATCH_SIZE = 8
EPOCHS = 10
LATENT_DIM = 3

if __name__ == '__main__':
    devices = jax.local_devices()
    num_devices = len(devices)
    sharding = jax.sharding.PositionalSharding(devices)
    assert BATCH_SIZE % num_devices == 0
    print(f"Found {num_devices} devices")

    # Prepare data
    images, labels = load_mnist("train")
    class_buckets = organize_by_class(images, labels)
    # Test the triplet sampling
    rng = random.PRNGKey(0)
    rng = jax.device_put(rng, sharding.replicate())
    anchor_img, positive_img, negative_img = get_triplet(
        class_buckets, rng, batch_size=BATCH_SIZE)
    assert anchor_img.shape == positive_img.shape == negative_img.shape

    # Initialize model
    key = random.PRNGKey(0)
    model = make_embodinet(type=MODEL_TYPE, latent_dim=LATENT_DIM)

    # replicate model across devices TODO: fix this
    # sharded_model = [jax.tree_map(lambda x: x[i], model) for i in range(num_devices)]
    # model = jax.device_put_sharded(sharded_model, jax.local_devices())

    if MODEL_TYPE == "conv":
        input_shape = (1, 28, 28, 1)
        keys = {'params': key}
    if MODEL_TYPE == "transformer":
        input_shape = (1, 784, 1)  # batch_size, seq_len, embed_dim
        batch_size, seq_len, embed_dim = input_shape
        # raise NotImplementedError("TODO: transformer")
        init_rng, dropout_rng = random.split(key, 2)
        keys = {'params': init_rng, 'dropout': dropout_rng}

    params = model.init(keys, jnp.ones(input_shape))['params']

    # Define the optimizer
    optimizer = optax.adam(learning_rate=1e-3)

    # Training state
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    # Training step
    def train_step(state, batch, rngs):
        def loss_fn(params):
            anchors, positives, negatives = batch
            anchor_embeddings = state.apply_fn(
                {'params': params}, anchors, rngs=rngs)
            positive_embeddings = state.apply_fn(
                {'params': params}, positives, rngs=rngs)
            negative_embeddings = state.apply_fn(
                {'params': params}, negatives, rngs=rngs)
            loss = triplet_loss(anchor_embeddings,
                                positive_embeddings, negative_embeddings)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss

    # Dummy data for illustration purposes only
    # Replace this with real triplet generation logic
    # anchors = jnp.ones((BATCH_SIZE, 28, 28, 1))
    # positives = jnp.ones((BATCH_SIZE, 28, 28, 1))
    # negatives = jnp.zeros((BATCH_SIZE, 28, 28, 1))

    # Training loop
    for epoch in range(EPOCHS):
        # for batch in train_ds:
        # Here, you would extract anchor, positive and negative images from the batch
        # For now, we'll use the dummy data
        rng, key = random.split(rng)
        anchors, positives, negatives = get_triplet(
            class_buckets, key, batch_size=BATCH_SIZE)

        if MODEL_TYPE == "transformer":
            # Flatten the images for transformer
            anchors = anchors.reshape(-1, 784, 1)
            positives = positives.reshape(-1, 784, 1)
            negatives = negatives.reshape(-1, 784, 1)

        state, loss = train_step(
            state, (anchors, positives, negatives), rngs={'dropout': rng})
        print(f'Epoch {epoch} done', loss)

    print('Training complete.')

    # Evaluate the model
    # Load test data
    images, labels = load_mnist("test")

    # get only 1 and 2 image per class
    class_buckets = organize_by_class(images, labels)

    first_10_digits = []
    second_10_digits = []
    for i in range(10):
        first_10_digits.append(class_buckets[i][0])
        second_10_digits.append(class_buckets[i][1])

    rng = random.PRNGKey(0)

    # Get embeddings for the test images
    first_10_digits = jnp.stack(first_10_digits)
    second_10_digits = jnp.stack(second_10_digits)

    if MODEL_TYPE == "transformer":
        first_10_digits = first_10_digits.reshape(-1, 784, 1)
        second_10_digits = second_10_digits.reshape(-1, 784, 1)

    print(first_10_digits.shape, second_10_digits.shape)
    first_10_embeddings = state.apply_fn(
        {'params': state.params}, first_10_digits, rngs={'dropout': rng})
    second_10_embeddings = state.apply_fn(
        {'params': state.params}, second_10_digits, rngs={'dropout': rng})

    # print out a 5 x 5 table of distances between embeddings
    print("Distance between embeddings")
    print("---------------------------")
    print("   ", end="")
    for i in range(10):
        print(f"   {i}  ", end="")
    print()
    for i in range(10):
        print(f"{i}  ", end="")
        for j in range(10):
            dist = jnp.sum(jnp.square(
                first_10_embeddings[i] - second_10_embeddings[j]))
            print(f"{dist:.2f}  ", end="")
        print()
    print()

    # matplot lib show 2 rows of 10 digit images
    fig, axs = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(10):
        axs[0, i].imshow(first_10_digits[i].reshape(28, 28))
        axs[1, i].imshow(second_10_digits[i].reshape(28, 28))
    plt.show()
