#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import os
import random
import jax.numpy as jnp

# This is similar to tf.config.experimental.set_memory_growth,
# which sets the GPU memory growth as needed basis to prevent OOM
# https://www.tensorflow.org/guide/gpu
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def pretty_list(x): return [round(float(i), 2) for i in x]


def img_preprocess(img):
    # Normalize the image to [0, 1]
    return img / 255.0


class TrajectoryDataset:
    def __init__(
        self,
        data_directory: str,
        use_npy: bool = True,
        traj_size: int = 15,
        random_skip: bool = False,
        max_traj_size: int = 15,
        random_traj_sampling: bool = True,
    ):
        """
        Simple script to create a tf.data.Dataset from a directory of .npy files.

        The dataset is as such (with batch size 15)
        {
            'images': np.array([15, 64, 64, 3]),
            'states': np.array([15, 7]),        # xyz, rpy, gripper
            'actions': np.array([15, 7]),       # delta xyz, rpy, gripper
            'cam_profile': np.array([7])        # cx, cy, cz, tx, ty, tz, fovy
        }
        """
        self.data_directory = data_directory
        self.traj_size = traj_size
        self.random_skip = random_skip
        self.max_traj_size = max_traj_size
        self.random_traj_sampling = random_traj_sampling

        if not use_npy:
            raise NotImplementedError


    def _trajectory_generator(self):
        # List all .npy files in the data directory
        files = [f for f in os.listdir(
            self.data_directory) if f.endswith('.npy')]
        random.shuffle(files)

        # hacky way to have a max skip
        _skip_count = 0

        for file in files:
            # Load the data from the .npy file
            data = np.load(os.path.join(
                self.data_directory, file), allow_pickle=True)

            # Iterate over the trajectories in the data
            for traj in data:
                if len(traj) < self.traj_size:
                    raise ValueError(
                        f"Trajectory length is less than {self.traj_size}")

                # A hacky way to impl skip of the current trajectory
                # 50% chance to skip the trajectory
                if self.random_skip and random.choice([True, False]):
                    _skip_count += 1
                    if _skip_count < 4:
                        continue
                    _skip_count = 0

                width, height, channels = traj[0]['observation']['image'].shape
                states_size = traj[0]['observation']['state'].shape[0]
                images = np.zeros((self.traj_size, width, height, channels))
                states = np.zeros((self.traj_size, states_size))
                actions = np.zeros((self.traj_size, 7))
                cam_profile = traj[0]['observation']['info/pixels_profile']

                # Randomly sample a trajectory of size self.traj_size in 
                # the current trajectory
                if self.random_traj_sampling:
                    start_idx = random.randint(
                        0, len(traj) - self.traj_size - 1)
                    traj = traj[start_idx:start_idx + self.traj_size]

                # Iterate over the transitions in the trajectory
                for i, transition in enumerate(traj):
                    if i == self.traj_size:
                        break
                    images[i] = img_preprocess(transition['observation']['image'])
                    states[i] = transition['observation']['state']
                    actions[i] = transition['action']

                # Yield the image and state as a dictionary
                yield {
                    'images': images,
                    'states': states,
                    'actions': actions,
                    'cam_profile': cam_profile,
                }

    def get_dataset(self):
        # Define the output types and shapes
        output_types = {
            'images': tf.float32,
            'states': tf.float32,
            'actions': tf.float32,
            'cam_profile': tf.float32
        }
        output_shapes = {
            'images': (None, None, None, 3),
            'states': (None, 7),
            'actions': (None, 7),
            'cam_profile': (None,)
        }

        # Create the dataset from the generator
        dataset = tf.data.Dataset.from_generator(
            self._trajectory_generator,
            output_types=output_types,
            output_shapes=output_shapes
        )
        return dataset


##############################################################################


def calc_norm_info(
    data_iter,
    feature_keys=['cam_profile', 'states', 'actions'],
    type='jnp',
):
    """
    Return the normalization info for the dataset.

    format is: { key_str: (min, max)}
    """
    norm_info = {}
    data = next(data_iter)

    for f in feature_keys:
        if len(data[f].shape) == 2:
            axis = 0
        elif len(data[f].shape) == 3:
            axis = (0, 1)
        else:
            raise ValueError(f"unsupported shape for {f} in calc_norm_info")

        # init norm info with (norm_min, norm_max)
        norm_info[f] = (
            tf.reduce_min(data[f], axis=axis),
            tf.reduce_max(data[f], axis=axis)
        )
        for i in range(100):
            data = next(data_iter)
            norm_info[f] = (
                tf.minimum(norm_info[f][0], tf.reduce_min(data[f], axis=axis)),
                tf.maximum(norm_info[f][1], tf.reduce_max(data[f], axis=axis))
            )

        # convert values to jnp or numpy, default is tf
        if type == 'jnp':
            norm_info[f] = (jnp.array(norm_info[f][0]),
                            jnp.array(norm_info[f][1]))
        elif type == 'np':
            norm_info[f] = (norm_info[f][0].numpy(), norm_info[f][1].numpy())

        norm_min, norm_max = norm_info[f]
        print(
            f"[Norm info] {f}: min {pretty_list(norm_min)}, max {pretty_list(norm_max)}")

    return norm_info


def normalize(x, norm_min_max):
    """Normalize data to [0, 1] range."""
    min_val, max_val = norm_min_max
    return (x - min_val) / (max_val - min_val)


def denormalize(x, norm_min_max):
    """Denormalize data from [0, 1] range."""
    min_val, max_val = norm_min_max
    return x * (max_val - min_val) + min_val


##############################################################################
if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Dataset')
    parser.add_argument('--data_path', '-d', type=str, help='Path to dataset')
    args = parser.parse_args()

    td = TrajectoryDataset(args.data_path, traj_size=6, random_skip=True)
    dataset = td.get_dataset()
    dataset = dataset.repeat()

    batched_dataset = dataset.batch(16)

    # use iterator to get the data
    iterator = iter(batched_dataset)

    # Get the normalization info
    norm_info = calc_norm_info(iterator)

    for i in range(10):
        print(f"Batch {i}")

        try:
            data = next(iterator)

            # Normalize the data
            norm_states = normalize(
                jnp.array(data['states']), norm_info['states'])
            norm_actions = normalize(data['actions'], norm_info['actions'])
            norm_cam_profile = normalize(
                data['cam_profile'], norm_info['cam_profile'])

            # pick batch index 0 to print
            print(f" States: {pretty_list(norm_states[0][0])}")
            print(f" Actions: {pretty_list(norm_actions[0][0])}")
            print(f" Cam Profile: {pretty_list(norm_cam_profile[0])}")

        except StopIteration:
            print("No more items in iterator")
            break

    print("Done")
