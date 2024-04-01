
import flax.linen as nn
import jax.numpy as jnp
import jax
from flax import linen as nn
from transformer import PositionalEncoding
import functools as ft


class FilmConditioning(nn.Module):
    """
    Film is adapted from https://github.com/google-research/robotics_transformer/blob/master/film_efficientnet/film_conditioning_layer.py
    original paper: https://arxiv.org/pdf/1709.07871.pdf
    """
    @nn.compact
    def __call__(self, conv_filters: jnp.ndarray, conditioning: jnp.ndarray):
        """Applies FiLM conditioning to a convolutional feature map.

        Args:
            conv_filters: A tensor of shape [batch_size, height, width, channels].
            conditioning: A tensor of shape [batch_size, conditioning_size].

        Returns:
            A tensor of shape [batch_size, height, width, channels].
        """
        projected_cond_add = nn.Dense(
            features=conv_filters.shape[-1],
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(conditioning)
        projected_cond_mult = nn.Dense(
            features=conv_filters.shape[-1],
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(conditioning)

        projected_cond_add = projected_cond_add[:, None, None, :]
        projected_cond_mult = projected_cond_mult[:, None, None, :]

        return conv_filters * (1 + projected_cond_add) + projected_cond_mult


class FilmConditioning3D(nn.Module):
    @nn.compact
    def __call__(self, conv_filters: jnp.ndarray, conditioning: jnp.ndarray):
        """Applies FiLM conditioning to a 3D convolutional feature map.

        Args:
            conv_filters: A tensor of shape [batch_size, time, height, width, channels].
            conditioning: A tensor of shape [batch_size, time, conditioning_size].

        Returns:
            A tensor of shape [batch_size, time, height, width, channels].

        # NOTE (YL): this is modified from above 2D version
        """
        projected_cond_add = nn.Dense(
            features=conv_filters.shape[-1],
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(conditioning)
        projected_cond_mult = nn.Dense(
            features=conv_filters.shape[-1],
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(conditioning)

        projected_cond_add = projected_cond_add[:, :, None, None, :]
        projected_cond_mult = projected_cond_mult[:, :, None, None, :]

        return conv_filters * (1 + projected_cond_mult) + projected_cond_add


class Conv3DEncoder(nn.Module):
    """
    Simple 3D Convolutional Encoder
    """
    latent_dim: int = 256
    use_film: bool = False

    @nn.compact
    def __call__(self, x, cond=None, is_training=True):
        """Cond is used when FiLM conditioning is enabled."""
        # 2 conv, max_and 2 dense
        x = nn.Conv(features=16, kernel_size=(3, 3, 3))(x)
        x = nn.BatchNorm(use_running_average=not is_training)(x)
        if self.use_film:
            x = FilmConditioning3D()(x, cond)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3, 3))(x)
        x = nn.BatchNorm(use_running_average=not is_training)(x)
        if self.use_film:
            x = FilmConditioning3D()(x, cond)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2, 2), strides=(2, 2, 2))
        # flatten
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.latent_dim)(x)
        return x


class PatchEncoder(nn.Module):
    """Takes an image and breaks it up into patches of size (patch_size x patch_size),
    applying a fully connected network to each patch individually.

    The default "encoder" used by most ViTs in practice.
    """

    use_film: bool = False
    patch_size: int = 32
    num_features: int = 256

    @nn.compact
    def __call__(self, observations: jnp.ndarray, train: bool = True):
        x = observations
        x = nn.Conv(
            features=self.num_features,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            name="embedding",
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        return x


class FlattenPatchEncoder(nn.Module):
    output_dim: int = 256
    patch_size: int = 16
    
    @nn.compact
    def __call__(self, x, is_training: bool = False):
        x = PatchEncoder(patch_size=self.patch_size, num_features=self.output_dim)(
            x, train=is_training
        )
        # completely flatten the patches
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        # x = x.reshape((x.shape[0], -1))
        raise NotImplementedError("Not implemented yet")
        return x


class SimpleViTEncoder(nn.Module):
    num_classes: int
    hidden_dim: int = 768
    num_heads: int = 6
    num_layers: int = 6
    mlp_dim: int = 1024
    patch_size: int = 16
    patch_features: int = 256

    @nn.compact
    def __call__(self, x, is_training: bool = False):
        # Patch Encoder
        x = PatchEncoder(patch_size=self.patch_size, num_features=self.patch_features)(
            x, train=is_training
        )

        x = x.reshape(x.shape[0], -1, x.shape[-1])  # Flatten patches
        x = PositionalEncoding(d_model=self.patch_features)(x)

        # Transformer blocks
        for _ in range(self.num_layers):
            y = nn.LayerNorm()(x)
            # use multi-head dot-product attention as nn.SelfAttention is deprecated
            y = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.hidden_dim,
                kernel_init=nn.initializers.xavier_uniform(),
                deterministic=not is_training, # set to True for evaluation
            )(x, x)
            x = x + y
            x = nn.LayerNorm()(x)
            y = nn.Dense(features=self.mlp_dim)(x)
            y = nn.relu(y)
            y = nn.Dense(features=self.patch_features)(y)
            y = nn.Dropout(rate=0.1)(y, deterministic=not is_training)
            x = x + y

        # output embedding
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=self.num_classes)(x)
        return x


vit_encoder_configs = {
    "mini-vit": ft.partial(
        SimpleViTEncoder,
        mlp_dim=1024,
    ),
    "small-vit": ft.partial(
        SimpleViTEncoder,
        mlp_dim=2048,
    ),
}


if __name__ == "__main__":
    import os
    # explicitly set the memory allocation to avoid OOM
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".20"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    model = vit_encoder_configs["small-vit"](num_classes=512)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 6, 128, 128, 3)))
    
    # model = Conv3DEncoder(latent_dim=256)
    # params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 12, 128, 128, 3)), None)

    # print number of parameters
    param_count = sum(p.size for p in jax.tree_util.tree_flatten(params)[0])
    print("Number of parameters: ", param_count)
    # approx size of the model in MB
    print("Model size (MB): ", param_count * 4 / 1024 / 1024)
    print("done")
