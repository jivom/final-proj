import numpy as np
import matplotlib.pyplot as plt
import cv2

background = None

def set_img(image):
    global background
    background = image

def load_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    small = cv2.resize(img, (0,0), fx=0.1, fy=0.1)
    
    return img, small
# code from tutorial with a small adjustment for our project
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box):
    plt.figure(figsize=(10,10))
    plt.imshow(background)
    ax = plt.gca()
    x0, y0 = box[0], box[1]
    w, h = box[2] , box[3] 
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  
    plt.show()
    
def show_anns(anns):
    plt.figure(figsize=(10,10))
    plt.imshow(background)
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)  
    plt.show()  
    
    
def hightlight_object(ann):
    img_copy = background.copy()
    mask = ann['segmentation']
    
    img_copy[mask] = img_copy[mask] * 0.5 + np.array([255, 0, 0]) * 0.5
    plt.imshow(img_copy)
    plt.show()