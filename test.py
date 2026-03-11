import os

os.system("CUDA_VISIBLE_DEVICES=0 python mmdetection/tools/test.py  ./mmdetection/configs/cascade_rcnn/cascade-mask-rcnn_r101_fpn_1x_coco.py  ./mmdetection/weight/instance_seg/cascade_mask_rcnn_r101_fpn_1x_coco_20200203-befdf6ee.pth  --work-dir ./results")
