# 2024 SNU 3D CV project

Our code is base of nerfmm : https://github.com/ActiveVisionLab/nerfmm/tree/main

# Updated
- 2024.06.05 Data generator(Blender format) updated ( see generator branch)
- 2024.06.05 model now can process accept file format ( see /dataloader/with_blender.py & tasks/*.py )
- 2024.06.06 Add nerf ( We have to test it) (see tasks/*.py )

# To Do..

- CFNerf, Nerf-four, mc-nerf? etc..
- Data
- Dust3r

# How to use

## Train
use train.py

    python tasks/nerfmm/train.py \
    --base_dir='path of data directory' \
    --scene_name='choodse scene' \
    --dtype='data formate type' \
    --model='model name'
    
## Evaluation
### Compute image quality matrix
- Eval.py

    ```
    python tasks/nerfmm/eval.py \
    --base_dir='path of data directory' \
    --scene_name='choodse scene' \
    --dtype='data formate type' \
    --ckpt_dir='path of checkpoints directory' \
    --model='model name'
    ```

### Rendering novel vies
- spiral.py

    ```
    python tasks/nerfmm/spiral.py \
    --base_dir='path of data directory' \
    --scene_name='choodse scene' \
    --dtype='data formate type' \
    --ckpt_dir='path of checkpoints directory' \
    --model='model name'
    ```

### Visualise estimated pose in 3D
- vis_learned_poses.py

    ```
    python tasks/nerfmm/vis_learned_poses.py \
    --base_dir='path of data directory' \
    --scene_name='choodse scene' \
    --dtype='data formate type' \
    --ckpt_dir='path of checkpoints directory'
    ```

After vis_learned_poses, .ply file would be created. Use MeshLab to visualize
