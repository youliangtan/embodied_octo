import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import os


class TrajectoryDataset:
    def __init__(
            self,
            data_directory: str, 
            use_npy: bool = True,
            traj_size: int = 15
        ):
        """
        Simple script to create a tf.data.Dataset from a directory of .npy files.
        
        The dataset is as such
        {
            'images': np.array([15, 64, 64, 3]),
            'states': np.array([15, 7]),        # xyz, rpy, gripper
            'cam_profile': np.array([7])        # cx, cy, cz, tx, ty, tz, fovy
        }
        """
        self.data_directory = data_directory
        self.traj_size = traj_size

        if not use_npy:
            raise NotImplementedError

    def trajectory_generator(self, repeat: bool = False):
        # List all .npy files in the data directory
        files = [f for f in os.listdir(self.data_directory) if f.endswith('.npy')]


        for file in files:
            # Load the data from the .npy file
            data = np.load(os.path.join(self.data_directory, file), allow_pickle=True)

            # Iterate over the trajectories in the data
            for traj in data:
                if len(traj) < self.traj_size:
                    raise ValueError(f"Trajectory length is less than {self.traj_size}")

                width, height, channels = traj[0]['observation']['image'].shape
                states_size = traj[0]['observation']['state'].shape[0]
                images = np.zeros((self.traj_size, width, height, channels))
                states = np.zeros((self.traj_size, states_size))
                cam_profile = traj[0]['observation']['info/pixels_profile']

                # Iterate over the transitions in the trajectory
                for i, transition in enumerate(traj):
                    if i == self.traj_size:
                        break
                    images[i] = transition['observation']['image']
                    states[i] = transition['observation']['state']

                # Yield the image and state as a dictionary
                yield {
                    'images': images,
                    'states': states,
                    'cam_profile': cam_profile,
                }

    def get_dataset(self):
        # Define the output types and shapes
        output_types = {
            'images': tf.float32,
            'states': tf.float32,
            'cam_profile': tf.float32
        }
        output_shapes = {
            'images': (None, None, None, 3),
            'states': (None, None),
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
