import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
import optax
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt


class Encoder(nn.Module):
    latents: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(500, name='fc1')(x)
        x = nn.relu(x)
        mean_x = nn.Dense(self.latents, name='fc2_mean')(x)
        logvar_x = nn.Dense(self.latents, name='fc2_logvar')(x)
        return mean_x, logvar_x


class Decoder(nn.Module):

    @nn.compact
    def __call__(self, z):
        z = nn.Dense(500, name='fc1')(z)
        z = nn.relu(z)
        z = nn.Dense(784, name='fc2')(z)
        return z


class VAE(nn.Module):
    encoder: Encoder
    decoder: Decoder

    def __call__(self, x, z_rng):
        mean, logvar = self.encoder(x)
        z = reparameterize(z_rng, mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar


def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std


def model(latent_dim):
    return VAE(
        encoder=Encoder(latents=latent_dim),
        decoder=Decoder(),
    )


@jax.vmap
def kl_divergence(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


@jax.vmap
def binary_cross_entropy_with_logits(logits, labels):
    logits = nn.log_sigmoid(logits)
    return -jnp.sum(labels * logits + (1. - labels) * jnp.log(-jnp.expm1(logits)))


##############################################################################

if __name__ == '__main__':

    # Check if a GPU is available
    if jax.local_device_count('gpu') > 0:
        print('GPU is available!')
    else:
        print('GPU is not available.')

    # PARAMETERS
    BATCH_SIZE = 256
    LATENTS = 5
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 100
    steps_per_epoch = 50000 // BATCH_SIZE

    def prepare_image(x):
        x = tf.cast(x['image'], tf.float32)
        x = tf.reshape(x, (-1,))
        return x

    ds_builder = tfds.builder('binarized_mnist')
    ds_builder.download_and_prepare()
    train_ds = ds_builder.as_dataset(split=tfds.Split.TRAIN)
    train_ds = train_ds.map(prepare_image)
    train_ds = train_ds.cache()
    train_ds = train_ds.repeat()
    train_ds = train_ds.shuffle(50000)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = iter(tfds.as_numpy(train_ds))

    test_ds = ds_builder.as_dataset(split=tfds.Split.TEST)
    test_ds = test_ds.map(prepare_image).batch(10000)
    test_ds = np.array(list(test_ds)[0])
    test_ds = iter(test_ds)

    # Initialize the parameters and the optimizer using optax
    rng = random.PRNGKey(0)
    rng, key = random.split(rng)

    init_data = jnp.ones((BATCH_SIZE, 784), jnp.float32)
    model_instance = model(latent_dim=LATENTS)
    params = model_instance.init(key, init_data, rng)['params']

    # Create an Adam optimizer with optax
    optimizer = optax.adam(LEARNING_RATE)

    # Initialize the optimizer state.
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, batch, z_rng):
        def loss_fn(params):
            recon_x, mean, logvar = model_instance.apply(
                {'params': params}, batch, z_rng)
            bce_loss = binary_cross_entropy_with_logits(recon_x, batch).mean()
            kld_loss = kl_divergence(mean, logvar).mean()
            loss = bce_loss + kld_loss
            return loss, recon_x  # Note we're returning a tuple here

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        # Unpack the correct number of return values
        (loss, recon_x), grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss

    # print the full network structure tree
    print(model_instance)
    print('\nStarting training...')

    # Training loop
    for epoch in range(NUM_EPOCHS):
        for _ in range(steps_per_epoch):
            batch = next(train_ds)
            rng, key = random.split(rng)
            params, opt_state, loss = train_step(params, opt_state, batch, key)
            # Now you can use 'params' for further processing or evaluation
        print(f'Epoch {epoch+1} | Loss {loss:.3f}')

    print('Training finished, evaluating...')

    # test ds
    for i in range(2):
        # Create a figure with 2 columns and 8 rows
        fig, axs = plt.subplots(8, 2, figsize=(8, 16))

        for j in range(8):
            # use next test image from test_ds
            test_img = next(test_ds)
            rng, key = random.split(rng)
            recon_img, mean, logvar = model_instance.apply(
                {'params': params}, test_img, key)

            axs[j, 0].imshow(test_img.reshape(28, 28))
            axs[j, 1].imshow(recon_img.reshape(28, 28))
            print(" - recon image", j)

        # Adjust the subplot layout
        fig.tight_layout()

        # Show the plot
        plt.show()

    print('Done!')
