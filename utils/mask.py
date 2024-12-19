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


def find_object_on_table(masks, table_idx=0):
    
    # assume table is the largest mask
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    # table_ann = max(masks, key=(lambda x: x['area']))
    table_ann = sorted_masks[table_idx]

    bounding_box = table_ann['bbox']
    masks_on_table = []
    for ann in masks:
        if is_box_within(ann['bbox'], bounding_box) and np.any(ann['bbox'] != bounding_box):
            masks_on_table.append(ann)
    return (masks_on_table)

def find_object_given_table(table_bbox, masks):
    masks_on_table = []

    for mask in masks:
        y_coords, x_coords = np.where(mask["segmentation"])
        x = np.mean(x_coords)
        y = np.mean(y_coords)

        if table_bbox[0] <= x <= table_bbox[0] + table_bbox[2] and table_bbox[1] <= y <= table_bbox[1] + table_bbox[3]:
            masks_on_table.append(mask)
        



        # bounding_box = table_bbox
        # masks_on_table = []
        # for ann in masks:
        #     if is_box_within(ann['bbox'], bounding_box) and np.any(ann['bbox'] != bounding_box):
        #         masks_on_table.append(ann)
    return (masks_on_table)