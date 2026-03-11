import os
# os.system("python ./demo/demo.py --config-file configs/cityscapes/semantic-segmentation/a3fpn_mask2former_R50_bs16_90k.yaml --input /media/mengen/T7/public-data/CityScapes/leftImg8bit/val/lindau/lindau_000005_000019_leftImg8bit.png --output /media/mengen/T7/paper_figs --opts MODEL.WEIGHTS weight/a3fpn_mask2former_cityscapes_R50_bs16_120k_squeeze_1111_bottom_up_8gpus/model_best.pth")

# os.system("python ./demo/demo.py --config-file configs/cityscapes/semantic-segmentation/a3fpn_mask2former_R50_bs16_90k.yaml --input /media/mengen/T7/public-data/CityScapes/leftImg8bit/val/frankfurt/*.png --output /media/mengen/T7/paper_figs/A3FPN/cityscapes/a3fpn_mask2former  --opts MODEL.WEIGHTS weight/a3fpn_mask2former_cityscapes_R50_bs16_120k_squeeze_1111_bottom_up_8gpus/model_best.pth")

# os.system("python tools/visualize_data.py --source annotation --config-file  configs/cityscapes/semantic-segmentation/a3fpn_mask2former_R50_bs16_90k.yaml --output-dir /media/mengen/T7/paper_figs/A3FPN/cityscapes/ground_truth_0.5")

# os.system("python ./demo/demo.py --config-file configs/coco/instance-segmentation/a3fpn_mask2former_R50_bs16_50ep.yaml --input /home/mengen/Desktop/object_detection/*.jpg --output /media/mengen/T7/paper_figs/A3FPN/coco/instance_seg/a3fpn_mask2former  --opts MODEL.WEIGHTS weight/a3fpn_mask2former_coco_R50_bs16_50epoch_squeeze_1111_top_down_8gpus_extra_10ep/model_best.pth")
# os.system("python ./demo/demo.py --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml --input /home/mengen/Desktop/object_detection/*.jpg --output /media/mengen/T7/paper_figs/A3FPN/coco/instance_seg/mask2former  --opts MODEL.WEIGHTS weight/mask2former_r50_coco_instance_seg.pkl")
