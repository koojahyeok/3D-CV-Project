from dust3r import get_scene
from tqdm import tqdm

import os
import argparse

import numpy as np

def get_camera_params(data_dir, save_dir, scaler_fn=None):
    """
    return estimated camera parameter through DUSt3R
    """
    
    result = {}
    
    for img_name in os.listdir(data_dir):
        if img_name.endswith('.png'):
            img_path = os.path.join(data_dir, img_name)
            data = {'img_path': img_path}
            
            scene = get_scene(data_dir)
    
            intrinsic_matrix = scene.get_focals()
            extrinsic_matrix = scene.get_came()
            
            if scaler_fn:
                intrinsic_matrix[0, 0] = scaler_fn(intrinsic_matrix[0, 0])
                intrinsic_matrix[1, 1] = scaler_fn(intrinsic_matrix[1, 1])
                        
            result[img_name] = {
                'intrinsic_matrix': intrinsic_matrix,
                'extrinsic_matrix': extrinsic_matrix
            }
    # 알고리즘 넣기
    
    return result

def main(args):
    # 데이터 간단 처리
    scaler_fn = lambda x: 2 * x - 200
    camera_params = get_camera_params(args, scaler_fn=scaler_fn)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    for img_name, params in tqdm(camera_params.items(), 
                                 desc='estimating camera params...', 
                                 ncols=100, 
                                 total=len(camera_params.keys())):
        save_path = os.path.join(args.save_dir, img_name.replace('.png', '.npy'))
        np.save(save_path, params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/scene')
    parser.add_argument('--save_dir', type=str, default='data/scene')
    
    # add arguments 
    
    args = parser.parse_args()

    main(args)
    