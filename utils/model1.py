from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
# from sam2.build_sam import build_sam2
# from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
# import hydra

# hydra.core.global_hydra.GlobalHydra.instance().clear()
# hydra.initialize_config_module('sam2_configs', version_base='1.2')

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import torch

model = None
time_shown = False
SAM = "sam"
SAM2 = "sam2"

def time_show():
    global time_shown
    if not time_shown:
        time_shown = True
        print("Time will be shown")
        return
    time_shown = False
    print("Time will not be shown")
    
def load_model(type, model_path, sam_type=SAM):
    global model
    global time_shown
    time_shart = time.time()
    if model == None:
        print("Loading model")
    else:
        print("Reloading model")
    
    if sam_type == SAM:
        print("Using SAM")
        sam = sam_model_registry[type](checkpoint=model_path)
        if torch.cuda.is_available():
            sam.to(device="cuda")
        mask_generator = SamAutomaticMaskGenerator(sam)

    elif sam_type == SAM2:
        print("Using SAM2")
        config = 'sam2.1_hiera_t.yaml'
        sam2 = build_sam2(config, model_path, apply_postprocessing=False)
        if torch.cuda.is_available():
            sam2.to(device="cuda")
        mask_generator = SAM2AutomaticMaskGenerator(sam2)

    time_end = time.time()  
    print("Model loaded")
    if time_shown:
        print("Time taken to load model: ", time_end - time_shart)
    model = mask_generator



def mask_generate(img, sam_type=SAM):
    global model
    if model == None:
        print("Model not loaded, get Vit_b or SAM2 model now")
        if sam_type == SAM:
            load_model("Vit_b", "ckpt/vit_b.pth", sam_type)
        elif sam_type == SAM2:
            model_path = "ckpt/sam2.1_hiera_tiny.pt"
            load_model(None, model_path, sam_type)
    print("Generating mask")
    time_start = time.time()
    mask = model.generate(img)
    time_end = time.time()
    if time_shown:
        print("Time taken to generate mask: ", time_end - time_start)
    return mask


