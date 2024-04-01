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
python3 embodinet.py -d /hdd/mujoco_arms/14k_widowx_combine/ -b 30 -t 6 -e 350 -m mini-vit -n minivit-fixcam-rst-xl-14kds-reg2e-4-t6-lr1e-4-b30 -lr 0.0001 --l2_reg 0.0002 -rst -s /hdd/e_octo
```
