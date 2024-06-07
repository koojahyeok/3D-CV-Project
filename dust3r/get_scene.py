
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

import os

def get_scene(image_path, n_images, device='cuda', start_idx=0, interval=1, schedule = 'cosine'):
    batch_size = 1
    lr = 0.01
    niter = 300

    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"

    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # image_path = "/home/diya/Public/Image2Smiles/KMolOCR_DL_Server/supplementary/dust3r/llff_room/"
    # image_path = "/home/diya/Public/Image2Smiles/KMolOCR_DL_Server/supplementary/dust3r/llff_room/"
    is_img = lambda x: x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')
    
    image_files = os.listdir(image_path)
    image_files = [img_name for img_name in image_files if is_img(img_name)]
    image_files.sort()

    # images = load_images(['croco/assets/Chateau1.png', 'croco/assets/Chateau2.png'], size=512)
    print(image_files[start_idx:start_idx+n_images*interval:interval], "are selected.")
    images = load_images([os.path.join(image_path, img_name) for img_name in image_files[start_idx:start_idx+n_images]], size=512)
    
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)
    
    
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    # loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
    
    return scene

# for test

def get_scene_of_two(image_path, img1_idx, img2_idx, device='cuda', niter=10, schedule = 'cosine'):
    batch_size = 1
    lr = 0.01
    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"

    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # image_path = "/home/diya/Public/Image2Smiles/KMolOCR_DL_Server/supplementary/dust3r/llff_room/"
    # image_path = "/home/diya/Public/Image2Smiles/KMolOCR_DL_Server/supplementary/dust3r/llff_room/"
    is_img = lambda x: x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')
    
    image_files = os.listdir(image_path)
    image_files = [img_name for img_name in image_files if is_img(img_name)]
    image_files.sort()

    # images = load_images(['croco/assets/Chateau1.png', 'croco/assets/Chateau2.png'], size=512)
    # print(image_files[start_idx:start_idx+n_images*interval:interval], "are selected.")
    images = load_images([os.path.join(image_path, image_files[img1_idx]), os.path.join(image_path, image_files[img2_idx])], size=512)
    
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)
    
    
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
    
    return scene

def get_scene_from_indices(image_path, img1_idx, indices, device='cuda', niter=10, schedule = 'cosine'):
    batch_size = 1
    lr = 0.01
    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"

    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # image_path = "/home/diya/Public/Image2Smiles/KMolOCR_DL_Server/supplementary/dust3r/llff_room/"
    # image_path = "/home/diya/Public/Image2Smiles/KMolOCR_DL_Server/supplementary/dust3r/llff_room/"
    is_img = lambda x: x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')
    
    image_files = os.listdir(image_path)
    image_files = [img_name for img_name in image_files if is_img(img_name)]
    image_files.sort()

    # images = load_images(['croco/assets/Chateau1.png', 'croco/assets/Chateau2.png'], size=512)
    # print(image_files[start_idx:start_idx+n_images*interval:interval], "are selected.")
    images = load_images([os.path.join(image_path, image_files[img_idx]) for img_idx in [img1_idx] + indices], size=512)
    
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)
    
    
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
    
    return scene





