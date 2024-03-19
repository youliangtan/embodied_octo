# embodied_orca (Embodied Octo)

> Work in progress

Learning Cross Embodiments: Enhancing Robot Foundation Models with Embodiment Encoding

Dependencies:
 - Octo: https://github.com/octo-models/octo
 - Open embodiments dataset: https://github.com/google-deepmind/open_x_embodiment


Simple experimentation with mnist dataset
```bash
# internal configs
python3 embodiednet.py
```

Simple mujoco scripted policy for embodied constrastive learning model (clip like)
```bash
python embodied_clip.py -d DATASET_PATH
```
