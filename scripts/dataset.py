#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import os
import random

# This is similar to tf.config.experimental.set_memory_growth,
# which sets the GPU memory growth as needed basis to prevent OOM
# https://www.tensorflow.org/guide/gpu
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class TrajectoryDataset:
    def __init__(
        self,
        data_directory: str,
        use_npy: bool = True,
        traj_size: int = 15,
        random_skip: bool = False,
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

        if not use_npy:
            raise NotImplementedError

    def trajectory_generator(self):
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

                # Iterate over the transitions in the trajectory
                for i, transition in enumerate(traj):
                    if i == self.traj_size:
                        break
                    images[i] = transition['observation']['image']
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
            self.trajectory_generator,
            output_types=output_types,
            output_shapes=output_shapes
        )
        return dataset


##############################################################################
##############################################################################


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Dataset')
    parser.add_argument('--data_path', '-d', type=str, help='Path to dataset')
    args = parser.parse_args()

    td = TrajectoryDataset(args.data_path)
    dataset = td.get_dataset()
    # dataset = dataset.repeat()

    batched_dataset = dataset.batch(16)

    # use iterator to get the data
    iterator = iter(batched_dataset)

    for i in range(50):
        print(f"Batch {i}")

        try:
            data = next(iterator)
            # print(data.keys())
            # print(data['cam_profile'])
            # print(data['states'].shape)
        except StopIteration:
            print("No more items in iterator")
            break

    print("Done")
