from ultralytics import YOLO
import torch
import numpy as np

def get_largest(results):
    table_boxes = []
    table_areas = []
    predicted_boxes = results[0].boxes.xyxy

    for i, box in enumerate(results[0].boxes):
        if results[0].names[int(box.cls)] == 'dining table':
            table_coords = predicted_boxes[i].to(device='cpu')
            table_boxes.append(table_coords)   
            area = (table_coords[2] - table_coords[0]) * (table_coords[3] - table_coords[1])
            table_areas.append(area)

    return table_boxes[np.argmax(table_areas)]

def get_table(img, conf=0.25, cuda=True):
    if cuda:
        torch.cuda.set_device(0)

    model = YOLO('ckpt/yolov8s.pt')

    # run the model on the image
    results = model.predict(source=img, conf=conf)

    # read in the image for visualization
    curr_table = get_largest(results)
    rectangle_img = img[int(curr_table[1]):int(curr_table[3])+1,int(curr_table[0]):int(curr_table[2])+1]
    return rectangle_img, curr_table