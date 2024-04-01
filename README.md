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
python3 embodied_clip.py -d /hdd/mujoco_arms/sviewpoint_combine/ -b 50 -t 6 -e 350 -m transformer -n xl-14kds-reg1e-3-t6-lr1e-4-b50 -lr 0.0001 --l2_reg 0.001 -s /hdd/e_octo
```
