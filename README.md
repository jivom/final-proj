# final-proj
CS 543 Final Project: Focused object detection for tetraplegic people using Mask R-CNN segmentation and gaze estimation


Requirements to run
1. pip install git+https://github.com/facebookresearch/segment-anything.git. If you want to use SAM2, use: pip install git+https://github.com/facebookresearch/sam2.git. SAM2 requires a different environment than SAM (python 3.10+).
- For SAM2
  * Ensure python version >= 3.10
  * sudo git clone https://github.com/facebookresearch/sam2.git && cd sam2
  * pip install -e .
2. pip install -r requirements.txt
3. **Download for vit_b** here: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
4. **Download for yolo** here: https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt (To run with yolo, use main_yolo.ipynb notebook)
5. **Download for SAM2** here: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
6. Save downloads under ckpt folder to run. Make sure they match with the file nams in the ipynb notebook desired.