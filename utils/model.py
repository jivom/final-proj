from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

model = None
time_shown = False

def time_show():
    global time_shown
    if not time_shown:
        time_shown = True
        print("Time will be shown")
        return
    time_shown = True
    print("Time will not be shown")
    
def load_model(type, model_path):
    global model
    global time_shown
    time_shart = time.time()
    if model == None:
        print("Loading model")
    else:
        print("Reloading model")
    sam = sam_model_registry[type](checkpoint=model_path)
    mask_generator = SamAutomaticMaskGenerator(sam)
    time_end = time.time()  
    print("Model loaded")
    if time_shown:
        print("Time taken to load model: ", time_end - time_shart)
    model = mask_generator



def mask_generate(img):
    global model
    if model == None:
        print("Model not loaded, get Vit_b model now")
        load_model("Vit_b", "./vit_b.pth")
    print("Generating mask")
    time_start = time.time()
    mask = model.generate(img)
    time_end = time.time()
    if time_shown:
        print("Time taken to generate mask: ", time_end - time_start)
    return mask


