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
    
def show_anns(anns, save_imag = False, img_name = 'output.png'):
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
    if save_imag:
        plt.savefig("./output/" + img_name)
    else:
        
        plt.show()  
    
    
def hightlight_object(ann):
    img_copy = background.copy()
    mask = ann['segmentation']
    
    img_copy[mask] = img_copy[mask] * 0.5 + np.array([255, 0, 0]) * 0.5
    plt.imshow(img_copy)
    plt.show()

def get_mask_line(arrow_start, arrow_end, draw_on_image = False):
    x1, y1 = arrow_start
    x2, y2 = arrow_end 
    m = (y2 - y1) / (x2 - x1) 
    c = (y1 - m * x1 )* 0.1 #this is the rescale factor

    m_inv = 1 / m
    b_inv = - c / m

    mask_line = np.zeros((background.shape[0], background.shape[1]), dtype=bool)

    if y1 < y2:
        for y in range(int(y1/10),background.shape[0]):
            x = int(m_inv * y + b_inv)
            if x >= 0 and x < background.shape[1]:
                mask_line[y, x] = True
    else:
        for y in range(0, int(y1/10)):
            x = int(m_inv * y + b_inv)
            if x >= 0 and x < background.shape[1]:
                mask_line[y, x] = True

    if x1 < x2:
        for x in range(int(x1/10),background.shape[1]):
            y = int(m * x + c)
            if y >= 0 and y < background.shape[0]:
                mask_line[y, x] = True
    else:
        for x in range(0, int(x1/10)):
            y = int(m * x + c)
            if y >= 0 and y < background.shape[0]:
                mask_line[y, x] = True
    if draw_on_image:
        background[mask_line] = [255, 0, 0]
    return mask_line