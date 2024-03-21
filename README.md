# embodied_orca (Embodied Octo)

> Work in progress

Learning Cross Embodiments: Enhancing Robot Foundation Models with Embodiment Encoding

Dependencies:
 - Octo: https://github.com/octo-models/octo
 - Open embodiments dataset: https://github.com/google-deepmind/open_x_embodiment


## Usage

1. Generate dataset with: https://github.com/rail-berkeley/mujoco_manipulation

2. Train the model with the generated dataset:

```bash
python3 embodied_clip.py -d /hdd/mujoco_arms/widowx_shoe_pick_and_place_combined/ -b 16 -t 12 -e 100
```
