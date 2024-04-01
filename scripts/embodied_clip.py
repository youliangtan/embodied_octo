#!/usr/bin/env python3

# Grand TODO:
#     - [ ] Condition this to a simple BC model
#     - [ ] Implement random static transformation to the cam profile, action, states

import os
from typing import Any, Optional
import wandb

import jax
import jax.numpy as jnp
from jax import tree_util
from jax import random, grad
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
import numpy as np

from transformer import TransformerPredictor
from utils import Timer
from networks import Conv3DEncoder, SimpleViTEncoder, FlattenPatchEncoder
from dataset import TrajectoryDataset, calc_norm_info, \
    normalize, denormalize

# explicitly set the memory allocation to avoid OOM
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".20"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def print_yellow(x): return print("\033[33m", x, "\033[0m")


def pretty_list(x): return [round(float(i), 2) for i in x]


class TrajectoryEncoder(nn.Module):
    """
    trajectory encoder uses the Conv3DEncoder,
    and stack an action array of size, then MLP
    """
    img_latent_dim: int = 64
    encoding_dim: int = 16
    use_film: bool = False
    ignore_actions: bool = False

    @nn.compact
    def __call__(self, images, actions, is_training=True):
        # 3D Convolutional Encoder
        x = Conv3DEncoder(latent_dim=self.img_latent_dim, use_film=self.use_film)(
            images,
            cond=actions,
            is_training=is_training
        )

        if not self.ignore_actions:
            # flatten the action
            actions = actions.reshape((actions.shape[0], -1))
            x = jnp.hstack([x, actions])  # now concatenate x and action
        # MLP
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.encoding_dim)(x)
        return x


class TrajectoryAttentionEncoder(nn.Module):
    """
    trajectory encoder uses the Conv3DEncoder,
    and stack an action array of size, then a transformer
    """
    img_latent_dim: int = 256
    encoding_dim: int = 16
    use_film: bool = True
    vit_encoder: Optional[SimpleViTEncoder] = None
    ignore_actions: bool = False

    @nn.compact
    def __call__(self, images, actions, is_training=True):
        # vision encoder
        if self.vit_encoder is not None:
            x = self.vit_encoder(
                images,
                is_training=is_training,
            )
        else:
            x = Conv3DEncoder(latent_dim=self.img_latent_dim, use_film=self.use_film)(
                images,
                cond=actions,
                is_training=is_training,
            )
        if not self.ignore_actions:
            # flatten the action
            actions = actions.reshape((actions.shape[0], -1))
            x = jnp.hstack([x, actions])
        # reshape the [batch, seq] to [batch, seq, features] with features = 1
        x = x.reshape((x.shape[0], x.shape[1], 1))
        x = TransformerPredictor(
            num_layers=5,
            model_dim=256,
            num_classes=self.encoding_dim,
            num_heads=4,
            dropout_prob=0.15,
            input_dropout_prob=0.05
        )(x, return_embedding=True, train=is_training)
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


def validate_model(
    test_ds_iter, model, train_state, norm_info,
    num_samples=15, rngs={'dropout': random.PRNGKey(0)}
):
    """Validate the model on the test dataset."""
    diff = jnp.zeros((7))
    diff_norm = jnp.zeros((7))
    mse = 0
    sample_labels, sample_preds = None, None  # hack to get the last sample
    for i in range(num_samples):
        data = next(test_ds_iter)
        # Convert TensorFlow tensors to jnp arrays
        data = {k: jnp.array(v) for k, v in data.items()}

        norm_states = normalize(data['states'], norm_info['states'])
        norm_actions = normalize(data['actions'], norm_info['actions'])
        state_actions = jnp.concatenate([norm_states, norm_actions], axis=2)

        cam_profile = data['cam_profile']
        norm_cam_profile = normalize(cam_profile, norm_info['cam_profile'])
        preds = model.apply(
            {'params': train_state.params, 'batch_stats': train_state.batch_stats},
            data['images'],
            state_actions,
            is_training=False,
            mutable=False,
            rngs=rngs,
        )
        preds_denorm = denormalize(preds, norm_info['cam_profile'])
        diff += jnp.mean(jnp.abs(preds_denorm - cam_profile), axis=0)
        diff_norm += jnp.mean(jnp.abs(preds - norm_cam_profile), axis=0)
        mse += jnp.mean(jnp.square(preds - norm_cam_profile))

        sample_labels, sample_preds = cam_profile[0], preds_denorm[0]

    diff /= num_samples
    diff_norm /= num_samples
    print(
        f" - Val | Average Diff: {pretty_list(diff_norm)}, with avg mse: {mse/num_samples}")

    wandb.log({"average_test_mse": mse/num_samples})
    for i in range(len(diff_norm)):
        wandb.log({f"diff_norm_{i}": diff_norm[i]})

    # cherrypick one of the sample to log
    # print it the array with round 2 decimal
    pred_vec_size = float(jnp.linalg.norm(sample_preds[3:6]))
    print("Sample labels and preds", pretty_list(sample_labels),
          pretty_list(sample_preds), pred_vec_size)

    # print normalized values
    print(
        "Sample norm labels and preds",
        pretty_list(normalize(sample_labels, norm_info['cam_profile'])),
        pretty_list(normalize(sample_preds, norm_info['cam_profile']))
    )


def baseline_predictor(test_ds_iter, norm_info, num_samples=10, random_pred=False):
    """Random predictor for the camera profile."""
    mse = 0
    avg_preds = jnp.zeros((7))
    for i in range(num_samples):
        data = next(test_ds_iter)
        cam_profile = jnp.array(data['cam_profile'])
        norm_cam_profile = normalize(cam_profile, norm_info['cam_profile'])

        if random_pred:
            # if prediction is uniform random from 0 to 1
            preds = random.uniform(key, shape=cam_profile.shape)
        else:
            # preds with middle value
            preds = jnp.ones(cam_profile.shape) * 0.5

        avg_preds += jnp.mean(preds, axis=0)
        mse += jnp.mean(jnp.square(preds - norm_cam_profile))

    mse /= num_samples
    avg_preds /= num_samples
    print(f"Random Predictor | Average MSE: {mse}")
    print(f"Random Predictor | Average Preds: {pretty_list(avg_preds)}")


def get_dataset_iter(dataset_dir, batch_size, traj_length, random_static_transform):
    td = TrajectoryDataset(dataset_dir, traj_size=traj_length,
                           random_static_transform=random_static_transform)
    dataset = td.get_dataset()
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset_iter = iter(dataset)

    # check the shape of the data
    datapoint = next(dataset_iter)
    images_shape = datapoint['images'].shape
    states_shape = datapoint['states'].shape
    actions_shape = datapoint['actions'].shape
    cam_profile_shape = datapoint['cam_profile'].shape
    assert len(images_shape) == 5, "images shape: (batch, time, h, w, c)"
    assert len(states_shape) == 3, "States shape: (batch, time, state_dim)"
    assert len(actions_shape) == 3, "Actions shape: (batch, time, action_dim)"
    assert len(
        cam_profile_shape) == 2, "Cam profile shape: (batch, cam_profile_dim)"
    return dataset_iter


def visualize_datapoint(data, preds, gt):
    """This simply show the images and the camera profile in 3D."""
    import matplotlib.pyplot as plt

    # visualize all images in a single row
    images = data['images'][0]
    images_count = images.shape[0]
    fig, axs = plt.subplots(1, images_count, figsize=(20, 5))
    for i in range(images_count):
        axs[i].imshow(images[i])
        axs[i].axis('off')

    pred_pt, pred_vec = preds[0][:3], preds[0][3:6]
    gt_pt, gt_vec = gt[0][:3], gt[0][3:6]

    # plot arrows of both pred and gt
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')

    ax.quiver(*pred_pt, *pred_vec, color='r', label='Predicted')
    ax.quiver(*gt_pt, *gt_vec, color='g', label='Ground Truth')

    base_link = data['base_link'][0]
    assert len(base_link) == 3, "Base link shape: (3,)"

    # draw the base link with huge arrow pointing in x direction
    ax.quiver(*base_link, 1, 0, 0, color='b', label='Base Link')

    bx, by, bz = base_link
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_xlim([bx - 1, bx + 1])
    ax.set_ylim([by - 1, by + 1])
    ax.set_zlim([bz - 1, bz + 1])
    ax.set_box_aspect([1, 1, 1])  # set axis aspect ratio
    ax.legend()
    plt.show()


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
    parser.add_argument('--model', '-m', type=str,
                        default="mlp", help='Model to use')
    parser.add_argument('--session_name', '-n', type=str,
                        default=None, help='Session name')
    parser.add_argument('--use_film', '-f', action='store_true')
    parser.add_argument('--save_path', '-s', type=str, default=None)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--l2_reg', '-l2', type=float, default=1e-3)
    parser.add_argument('--no_actions', '-na', action='store_true')
    parser.add_argument('--use_weighted_loss', '-wl', action='store_true')
    parser.add_argument('--validate_checkpoint', '-vc', type=str, default=None)
    parser.add_argument('--random_static_transform',
                        '-rst', action='store_true')
    args = parser.parse_args()

    train_dataset_dir = os.path.join(args.data_path, 'train')
    test_dataset_dir = os.path.join(args.data_path, 'val')

    train_ds_iter = get_dataset_iter(
        train_dataset_dir, args.batch_size, args.traj_length, args.random_static_transform)
    test_ds_iter = get_dataset_iter(
        test_dataset_dir, args.batch_size, args.traj_length, args.random_static_transform)

    devices = jax.local_devices()
    shard_count = len(devices)
    sharding = jax.sharding.PositionalSharding(devices)
    print(f"Devices: {devices}")

    # TODO: complete impl of sharding
    # create a 1D mesh with a single axis named "batch"
    mesh = Mesh(jax.devices(), axis_names="batch")
    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    data_parallel_sharding = NamedSharding(mesh, PartitionSpec("batch"))

    if args.model == "mlp":
        model = TrajectoryEncoder(
            img_latent_dim=256,
            encoding_dim=7,
            use_film=args.use_film,
            ignore_actions=args.no_actions
        )
    elif args.model == "transformer":
        model = TrajectoryAttentionEncoder(
            img_latent_dim=256,
            encoding_dim=7,
            use_film=args.use_film,
            ignore_actions=args.no_actions
        )
    elif args.model == "mini-vit":
        img_latent_dim = 512
        model = TrajectoryAttentionEncoder(
            img_latent_dim=img_latent_dim,
            encoding_dim=7,
            vit_encoder=SimpleViTEncoder(num_classes=img_latent_dim),
            use_film=args.use_film,
            ignore_actions=args.no_actions
        )
    elif args.model == "small-vit":
        img_latent_dim = 512
        model = TrajectoryAttentionEncoder(
            img_latent_dim=img_latent_dim,
            encoding_dim=7,
            vit_encoder=SimpleViTEncoder(num_classes=img_latent_dim,
                                         mlp_dim=2048,
                                         num_heads=8),
            use_film=args.use_film,
            ignore_actions=args.no_actions
        )
    elif args.model == "patch-only":
        img_latent_dim = 512
        model = TrajectoryAttentionEncoder(
            img_latent_dim=img_latent_dim,
            encoding_dim=7,
            vit_encoder=FlattenPatchEncoder(
                patch_size=16, output_dim=img_latent_dim),
            use_film=args.use_film,
            ignore_actions=args.no_actions
        )
    else:
        raise ValueError(f"Model {args.model} not implemented")

    optimizer = optax.adam(
        learning_rate=args.learning_rate, b1=0.9, b2=0.999, eps=1e-8,
    )

    # Sample first 100 data points to get the normalization range
    norm_info = calc_norm_info(train_ds_iter)

    @jax.jit
    def train_step(train_state, batch, rngs):
        def loss_fn(params):
            # forward pass,
            images, states, actions, cam_profile = \
                batch['images'], batch['states'], batch["actions"], batch['cam_profile']
            norm_states = normalize(states, norm_info['states'])
            norm_actions = normalize(actions, norm_info['actions'])

            state_actions = jnp.concatenate(
                [norm_states, norm_actions], axis=2)
            # state_actions = actions

            # TODO: now assume states as actions, need to change
            preds, updates = model.apply(
                {'params': params, 'batch_stats': train_state.batch_stats},
                images,
                state_actions,
                rngs=rngs,
                mutable=['batch_stats'],
            )

            # L2 regularization
            # https://github.com/google/flax/discussions/1654
            l2_reg_loss = 0.5 * sum(jnp.sum(jnp.square(p))
                                    for p in jax.tree_util.tree_leaves(params))

            # the loss will be L2 loss of the predicted and the true cam profile
            norm_cam_profile = normalize(cam_profile, norm_info['cam_profile'])
            denorm_preds = denormalize(preds, norm_info['cam_profile'])
            denorm_mae = jnp.mean(jnp.abs(denorm_preds - cam_profile))
            mse = jnp.mean(jnp.square(preds - norm_cam_profile))  # MSE

            # weighted loss
            if args.use_weighted_loss:
                lambda_w = jnp.array([5, 5, 5, 1, 1, 1, 5])
                loss = jnp.mean(
                    lambda_w * jnp.square(preds - norm_cam_profile))
            else:
                loss = mse
            loss += args.l2_reg * l2_reg_loss

            # loss += jnp.mean(jnp.abs(preds - norm_cam_profile)) # MAE
            # loss += jnp.sqrt(jnp.mean(jnp.square(preds - norm_cam_profile))) # rmse
            metrics = {'mse': mse, 'loss': loss, 'denorm_mae': denorm_mae}
            return loss, (metrics, updates)  # Return loss as the first output

        # compute the gradient
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (metrics, updates)), grad = grad_fn(
            train_state.params)  # Unpack the auxiliary data

        # update the train state
        train_state = train_state.apply_gradients(grads=grad)
        train_state = train_state.replace(
            params=train_state.params,
            opt_state=train_state.opt_state,
            batch_stats=updates['batch_stats']
        )
        return train_state, loss, metrics

    # initialize the model
    rng = random.PRNGKey(0)
    init_rng, dropout_rng = random.split(rng, 2)
    rngs = {'params': init_rng, 'dropout': dropout_rng}

    datapoint = next(train_ds_iter)
    imgs_shape = datapoint['images'].shape
    states_shape = datapoint['states'].shape
    actions_shape = datapoint['actions'].shape
    init_img_data = jnp.ones(
        (1, args.traj_length, imgs_shape[2], imgs_shape[3], imgs_shape[4]))
    init_act_data = jnp.ones(
        (1, args.traj_length, states_shape[2] + actions_shape[2]))

    variables = model.init(rngs,
                           init_img_data,
                           init_act_data,
                           is_training=False,
                           )
    params = variables['params']

    print(
        f"Total number of parameters: {count_params(params)}, "
        f"equivalent to {count_params(params) * 4 / 1024 / 1024} MB")

    # from batch borm
    # https://flax.readthedocs.io/en/latest/guides/training_techniques/batch_norm.html
    batch_stats = variables['batch_stats']

    class TrainState(train_state.TrainState):
        batch_stats: Any

    # opt_state = optimizer.init(params)
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        batch_stats=batch_stats,
        tx=optimizer,
    )

    # Validate the model with the checkpoint and visualize the predictions
    if args.validate_checkpoint:
        print_yellow(
            f"Validating the model with the checkpoint {args.validate_checkpoint}")
        train_state = checkpoints.restore_checkpoint(
            args.validate_checkpoint,
            target=train_state,
        )
        for i in range(99):
            data = next(test_ds_iter)
            data = {k: jnp.array(v) for k, v in data.items()}

            norm_states = normalize(data['states'], norm_info['states'])
            norm_actions = normalize(data['actions'], norm_info['actions'])
            state_actions = jnp.concatenate(
                [norm_states, norm_actions], axis=2)

            norm_preds = model.apply(
                {'params': train_state.params,
                    'batch_stats': train_state.batch_stats},
                data['images'],
                state_actions,
                is_training=False,
                rngs={'dropout': random.PRNGKey(0)}
            )

            preds = denormalize(norm_preds, norm_info['cam_profile'])
            norm_gt = normalize(data['cam_profile'], norm_info['cam_profile'])
            gt = data['cam_profile']

            print("Predictions", pretty_list(preds[0]))
            print("Ground Truth", pretty_list(gt[0]))
            print(" - MSE", jnp.mean(jnp.square(norm_preds - norm_gt)))

            visualize_datapoint(data, preds, gt)

        print("Done validating \n\n")
        exit()

    # init wandb
    wandb.init(project="embodied_octo",
               group="camera profile prediction",
               name=args.session_name
               )
    wandb.config.update(args)

    timer = Timer()
    ITERATIONS_PER_EPOCH = 100  # the number of iterations per epoch

    # training loop
    for epoch in range(args.num_epochs):
        avg_loss = 0
        avg_train_mse = 0
        avg_denorm_mae = 0
        for i in range(ITERATIONS_PER_EPOCH):
            rng, key = random.split(rng)

            with timer("timer/data_loading"):
                data = next(train_ds_iter)
            # Convert TensorFlow tensors to jnp arrays
            data = {k: jnp.array(v) for k, v in data.items()}

            with timer("timer/train_step"):
                train_state, loss, metrics = train_step(
                    train_state, data, rngs={'dropout': rng}
                )
            avg_loss += loss
            avg_train_mse += metrics['mse']
            avg_denorm_mae += metrics['denorm_mae']

        avg_loss /= ITERATIONS_PER_EPOCH
        avg_train_mse /= ITERATIONS_PER_EPOCH
        avg_denorm_mae /= ITERATIONS_PER_EPOCH
        print_yellow(
            f"Epoch {epoch} | Average Loss: {avg_loss} | Average MSE: {avg_train_mse} | Average Denorm MAE: {avg_denorm_mae}")

        # log after first epoch
        if epoch > 0:
            wandb.log(timer.get_average_times(reset=True))
        else:
            timer.reset()
        wandb.log({"average_loss": avg_loss, "average_train_mse": avg_train_mse,
                  "average_denorm_train_mae": avg_denorm_mae})
        validate_model(test_ds_iter, model, train_state, norm_info)

        # save the model every 50 epochs
        if args.save_path and epoch > 0 and epoch % 50 == 0:
            # support ~ and cwd for full path
            full_path = os.path.expanduser(args.save_path)
            full_path = os.path.join(full_path, args.session_name)

            # mkdir folder if not exist
            if not os.path.exists(full_path):
                os.makedirs(full_path)

            checkpoints.save_checkpoint(
                full_path, train_state, overwrite=True, step=epoch, keep=3,
            )
            print(f"Model saved to {full_path}")

    print("Done training")

    print("\n ---------- Validating Model ----------")
    baseline_predictor(test_ds_iter, norm_info)
    validate_model(test_ds_iter, model, train_state, norm_info)
    print("Done testing \n\n")
