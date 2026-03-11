# import os
# import torch
# Multiple gpus
# os.system('python3 ./detectron2/tools/train_net.py --config-file ./detectron2/configs/COCO-Detection/faster_rcnn_R_50_PRFPN_1x.yaml --num-gpus 2')
# os.system('python3 ./detectron2/tools/train_net.py --config-file ./detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_PRFPN_1x.yaml --num-gpus 2')
# os.system('python3 ./detectron2/tools/train_net.py --config-file ./detectron2/configs/COCO-PanopticSegmentation/panoptic_prfpn_R_50_1x.yaml --num-gpus 2')
# os.system('python train_net.py --config-file configs/cityscapes/semantic-segmentation/a3fpn_mask2former_R50_bs16_90k.yaml --num-gpus 8')
# os.system('python3 train_net.py --config-file configs/coco/instance-segmentation/a3fpn_mask2former_R50_bs16_50ep.yaml --num-gpus 8  --resume')
# os.system('python train_net.py --config-file configs/coco/instance-segmentation/a3fpn_mask2former_R50_bs16_extra_10ep.yaml --num-gpus 8')

# Single gpu
# os.system('python3 ./detectron2/tools/train_net.py --config-file ./detectron2/configs/COCO-Detection/faster_rcnn_R_50_PRFPN_1x.yaml --num-gpus 1')
# os.system('python3 ./detectron2/tools/train_net.py --config-file ./detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_PRFPN_1x.yaml --num-gpus 1')
# os.system('python3 ./detectron2/tools/train_net.py --config-file ./detectron2/configs/COCO-PanopticSegmentation/panoptic_prfpn_R_50_1x.yaml --num-gpus 1')
# os.system('python3 ./detectron2/tools/train_net.py --config-file ./detectron2/configs/Cityscapes/mask_rcnn_R_50_PRFPN.yaml --num-gpus 1')
# print(f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
