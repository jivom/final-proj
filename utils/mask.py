import numpy as np

def find_table(masks):
    # Find the largest mask
    largest_ann = max(masks, key=(lambda x: x['area']))
    return np.array([largest_ann])


def is_box_within(inner_box, outer_box):
    
    ix_min, iy_min, i_width, i_height = inner_box
    ox_min, oy_min, o_width, o_height = outer_box
    
    ix_max = i_width + ix_min
    iy_max = i_height + iy_min
    ox_max = o_width + ox_min
    oy_max = o_height + oy_min
    return (
        ix_min >= ox_min and ix_max <= ox_max and 
        iy_min >= oy_min and iy_max <= oy_max      
    )


def find_object_on_table(masks):
    
    # assume table is the largest mask
    table_ann = max(masks, key=(lambda x: x['area']))

    bounding_box = table_ann['bbox']
    masks_on_table = []
    for ann in masks:
        if is_box_within(ann['bbox'], bounding_box) and np.any(ann['bbox'] != bounding_box):
            masks_on_table.append(ann)
    return (masks_on_table)

def find_object_given_table(table_bbox, masks):
    bounding_box = table_bbox
    masks_on_table = []
    for ann in masks:
        if is_box_within(ann['bbox'], bounding_box) and np.any(ann['bbox'] != bounding_box):
            masks_on_table.append(ann)
    return (masks_on_table)