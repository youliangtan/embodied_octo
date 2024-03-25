# embodied_orca (Embodied Octo)

> Work in progress

Learning Cross Embodiments: Enhancing Robot Foundation Models with Embodiment Encoding

Dependencies:
 - Octo: https://github.com/octo-models/octo
 - Open embodiments dataset: https://github.com/google-deepmind/open_x_embodiment


## Usage

1. Generate dataset with: https://github.com/rail-berkeley/mujoco_manipulation

```bash
python data_collection/collect_scripted.py --task widowx_shoe_pick_and_place  -d /hdd/mujoco_arms/ -nt 2000 -t 15 -p 10
```

2. Train the model with the generated dataset:

```bash
python3 embodied_clip.py -d /hdd/mujoco_arms/sviewpoint_combine/ -b 20 -t 6 -e 250 -m transformer -n largetrans-b20-lr-2e-5-reg1e-4 -lr 0.00002
```
