# 2024 SNU 3D CV project

Our code is base of nerfmm : https://github.com/ActiveVisionLab/nerfmm/tree/main

# Have to do

- Basic nerf
- CFNerf, Nerf-four, mc-nerf? etc..
- Data
- Dust3r

# How to use

## Train
use train.py

    python tasks/nerfmm/train.py \
    --base_dir='path of data directory' \
    --scene_name='choodse scene' \
    --dtype='data formate type'
    
## Evaluation
### Compute image quality matrix
- Eval.py

    python tasks/nerfmm/eval.py \
    --base_dir='path of data directory' \
    --scene_name='choodse scene' \
    --dtype='data formate type' \
    --ckpt_dir='path of checkpoints directory'

### Rendering novel vies
- spiral.py

    python tasks/nerfmm/spiral.py \
    --base_dir='path of data directory' \
    --scene_name='choodse scene' \
    --dtype='data formate type' \
    --ckpt_dir='path of checkpoints directory'

### Visualise estimated pose in 3D
- vis_learned_poses.py

    python tasks/nerfmm/vis_learned_poses.py \
    --base_dir='path of data directory' \
    --scene_name='choodse scene' \
    --dtype='data formate type' \
    --ckpt_dir='path of checkpoints directory'

After vis_learned_poses, .ply file would be created. Use MeshLab to visualize
