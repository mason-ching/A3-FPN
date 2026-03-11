# import os
# import torch
# Multiple gpus
# os.system("CUDA_VISIBLE_DEVICES=0,1,2,3 ./mmdetection/tools/dist_train.sh a3fpn_mask-rcnn_r50_3x_coco-epoch.py 4 --work-dir ./weight/a3fpn_mask_rcnn_coco_3x_ms")

# os.system('python3 ./detectron2/tools/train_net.py --config-file ./detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml --num-gpus 4 --eval-only')
# os.system('python detectron2/tools/train_net.py --config-file detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_A3FPN_3x.yaml --num-gpus 1  --eval-only  --resume')
# os.system('python detectron2/tools/train_net.py --config-file detectron2/configs/PascalVOC-Detection/faster_rcnn_R_50_A3FPN.yaml --num-gpus 8  --eval-only')

# Single gpu
# os.system("CUDA_VISIBLE_DEVICES=0 python ./mmdetection/tools/train.py a3fpn_retinanet_r50_visdrone_1x.py --work-dir ./weight/retinanet_r50_a3fpn_visdrone_1x/")

