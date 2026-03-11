import argparse
import os
import shutil

import numpy as np
from osgeo import gdal
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
from mmseg.apis import init_model, inference_model


############################ 影像裁剪部分code #################################


#  读取tif数据集
def readTif(image_path):
    dataset = gdal.Open(image_path)
    if dataset is None:
        print(image_path + "文件无法打开")

    return dataset


#  保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if "int8" in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif "int16" in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(
        path, int(im_width), int(im_height), int(im_bands), datatype
    )
    if dataset is not None:
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


"""
滑动窗口裁剪函数
TifPath 影像路径
SavePath 裁剪后保存目录
CropSize 裁剪尺寸
RepetitionRate 重复率
"""


def TifCrop(TifPath, SavePath, CropSize, RepetitionRate, logger, infer_id, is_crop):
    dataset_img = readTif(TifPath)
    width = dataset_img.RasterXSize
    height = dataset_img.RasterYSize
    proj = dataset_img.GetProjection()
    geotrans = dataset_img.GetGeoTransform()
    if not is_crop:
        return width, height, proj, geotrans

    logger.info(f"width:{width}")
    logger.info(f"height:{height}")
    logger.info(f"proj:{proj}")
    logger.info(f"geotrans:{geotrans}")

    img = dataset_img.ReadAsArray(0, 0, width, height)  # 获取数据
    num_h = int(
        (height - CropSize * RepetitionRate) // (CropSize * (1 - RepetitionRate))
    )
    num_w = int(
        (width - CropSize * RepetitionRate) // (CropSize * (1 - RepetitionRate))
    )
    #  获取当前文件夹的文件个数len,并以len+1命名即将裁剪得到的图像
    new_name = len(os.listdir(SavePath)) + 1
    #  裁剪图片,重复率为RepetitionRate
    logger.info(
        "-------------------==================== Start Croping ======================---------------------"
    )

    for i in range(num_h):
        for j in range(num_w):
            #  如果图像是单波段
            if len(img.shape) == 2:
                cropped = img[
                          int(i * CropSize * (1 - RepetitionRate)): int(
                              i * CropSize * (1 - RepetitionRate)
                          )
                                                                    + CropSize,
                          int(j * CropSize * (1 - RepetitionRate)): int(
                              j * CropSize * (1 - RepetitionRate)
                          )
                                                                    + CropSize,
                          ]
            #  如果图像是多波段
            else:
                cropped = img[
                          :,
                          int(i * CropSize * (1 - RepetitionRate)): int(
                              i * CropSize * (1 - RepetitionRate)
                          )
                                                                    + CropSize,
                          int(j * CropSize * (1 - RepetitionRate)): int(
                              j * CropSize * (1 - RepetitionRate)
                          )
                                                                    + CropSize,
                          ]
            #  写图像
            writeTiff(cropped, geotrans, proj, f"{SavePath}/{infer_id}_{new_name}.tif")
            #  文件名 + 1
            new_name = new_name + 1
    logger.info(
        f"---------------- Normal range is complete. A total of {num_h * num_w} small block images！----------------"
    )

    #  向前裁剪最后一列
    for i in range(num_h):
        if len(img.shape) == 2:
            cropped = img[
                      int(i * CropSize * (1 - RepetitionRate)): int(
                          i * CropSize * (1 - RepetitionRate)
                      )
                                                                + CropSize,
                      (width - CropSize): width,
                      ]
        else:
            cropped = img[
                      :,
                      int(i * CropSize * (1 - RepetitionRate)): int(
                          i * CropSize * (1 - RepetitionRate)
                      )
                                                                + CropSize,
                      (width - CropSize): width,
                      ]
        #  写图像
        writeTiff(cropped, geotrans, proj, f"{SavePath}/{infer_id}_{new_name}.tif")
        new_name = new_name + 1
    logger.info(
        f"---------------- Rightmost column is complete. A total of {num_h} small block images！----------------"
    )

    #  向前裁剪最后一行
    for j in range(num_w):
        if len(img.shape) == 2:
            cropped = img[
                      (height - CropSize): height,
                      int(j * CropSize * (1 - RepetitionRate)): int(
                          j * CropSize * (1 - RepetitionRate)
                      )
                                                                + CropSize,
                      ]
        else:
            cropped = img[
                      :,
                      (height - CropSize): height,
                      int(j * CropSize * (1 - RepetitionRate)): int(
                          j * CropSize * (1 - RepetitionRate)
                      )
                                                                + CropSize,
                      ]
        writeTiff(cropped, geotrans, proj, f"{SavePath}/{infer_id}_{new_name}.tif")
        #  文件名 + 1
        new_name = new_name + 1
    logger.info(
        f"---------------- Bottom line is complete. A total of {num_w} small block images！----------------"
    )

    #  裁剪右下角
    if len(img.shape) == 2:
        cropped = img[(height - CropSize): height, (width - CropSize): width]
    else:
        cropped = img[:, (height - CropSize): height, (width - CropSize): width]
    # logger.info(f"---------------- Bottom right corner is complete. A total of {1} small block images！----------------")

    writeTiff(cropped, geotrans, proj, f"{SavePath}/{infer_id}_{new_name}.tif")
    new_name = new_name + 1

    logger.info(
        f"---------------- Crop complete! the output file is at {SavePath} ----------------"
    )

    return width, height, proj, geotrans


######################################### 影像拼接部分code #############################################

def stitchTiff(
        ori_img_path,
        croped_path,
        output_path,
        output_name,
        size,
        repetition,
        logger: logging.Logger,
        infer_id,
):
    ori_img = readTif(ori_img_path)

    croped_path = croped_path
    output_path = output_path
    output_name = output_name
    size = size
    repetition = repetition

    w = ori_img.RasterXSize
    h = ori_img.RasterYSize
    proj = ori_img.GetProjection()
    geotrans = ori_img.GetGeoTransform()
    num_h = (h - repetition) // (size - repetition)  # 裁剪后行数
    num_w = (w - repetition) // (size - repetition)  # 裁剪后列数
    img = np.zeros((h, w))  # 创建与原始图像等大的画布

    all_img = os.listdir(croped_path)  # ['1.tif', '10.tif', '100.tif', ...]
    all_img = [img for img in all_img if img.endswith(".tif")]
    all_img.sort(
        key=lambda x: int(x.split("_")[-1][:-4])
    )  # ['1.tif', '2.tif', '3.tif', ...]

    logger.info(
        "--------------------------------==============  Start Stitching ==============--------------------------------------"
    )

    # 1.正常范围拼接
    i, j = 0, 0
    for i in range(0, num_h):
        for j in range(0, num_w):
            small_img_path = os.path.join(croped_path, all_img[i * num_w + j])
            # print(f'正常范围拼接:{all_img[i * num_w + j]}')
            small_img = readTif(small_img_path)
            small_img = small_img.ReadAsArray(0, 0, size, size)  # 获取数据
            small_img = np.array(small_img)
            img[
            i * (size - repetition): i * (size - repetition) + size,
            j * (size - repetition): j * (size - repetition) + size,
            ] = small_img[0:size, 0:size]
    logger.info(
        f"---------------- Normal range is complete. A total of {num_w * num_h} small block images！----------------"
    )

    # 2.最右边一列的拼接
    for i in range(0, num_h):
        small_img_path = os.path.join(croped_path, all_img[num_h * num_w + i])
        # print(f'最右边一列的拼接:{all_img[num_h * num_w + i]}')
        small_img = readTif(small_img_path)
        small_img = small_img.ReadAsArray(0, 0, size, size)  # 获取数据
        small_img = np.array(small_img)
        img[i * (size - repetition): i * (size - repetition) + size, w - size: w] = (
            small_img[0:size, 0:size]
        )
    logger.info(
        f"---------------- Rightmost column is complete. A total of {num_h} small block images！----------------"
    )

    # 3.最下面一行的拼接:
    for j in range(0, num_w):
        small_img_path = os.path.join(croped_path, all_img[num_h * num_w + num_h + j])
        # print(f'最下面一行的拼接:{all_img[num_h * num_w + num_h + j]}')
        small_img = readTif(small_img_path)
        small_img = small_img.ReadAsArray(0, 0, size, size)  # 获取数据
        small_img = np.array(small_img)
        img[h - size: h, j * (size - repetition): j * (size - repetition) + size] = (
            small_img[0:size, 0:size]
        )
    logger.info(
        f"---------------- Bottom line is complete. A total of {num_w} small block images！----------------"
    )

    # 4.最右下角的一幅小图
    small_img_path = os.path.join(croped_path, all_img[-1])
    # print(f'最右下角的一幅小图拼接:{all_img[-1]}')
    small_img = readTif(small_img_path)
    small_img = small_img.ReadAsArray(0, 0, size, size)  # 获取数据
    small_img = np.array(small_img)
    img[h - size: h, w - size: w] = small_img[0:size, 0:size]
    logger.info(
        f"---------------- Bottom right corner is complete. A total of {1} small block images！----------------"
    )

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    writeTiff(img, geotrans, proj, os.path.join(output_path, output_name))

    logger.info(
        f"----------------============== Stitch complete! ==============----------------"
    )

    logger.info(
        f"============== the output file is at: [{os.path.join(output_path, output_name)}] =============="
    )


################################## 影像推理部分code ######################################


def check_img(image_path):
    if not (image_path.endswith(".tif", -4) or image_path.endswith(".TIF", -4)):
        raise TypeError(f"The type of input image must be in TIF format")

    dataset = gdal.Open(image_path)

    if dataset is None:
        raise FileNotFoundError("Unable to open the image for the path you entered!")

    projection = dataset.GetProjectionRef()
    geotransform = dataset.GetGeoTransform()

    if projection is None or geotransform is None:
        raise AttributeError(
            "The image file does not have a coordinate system or projection!"
        )

    dataset = None


def delete_dir(dir):
    try:
        shutil.rmtree(dir)
        print(f"path:[{dir}] had been deleted")
    except FileNotFoundError:
        print(f"path: [{dir}] is not exist")
    except Exception as e:
        print(f"delete path: [{dir}] happen error: [{str(e)}]")


def croptif(imgpath, save_path, cropsize, logger: logging.Logger, infer_id):
    check_img(imgpath)
    is_crop = False
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        logger.info(f"clip results save path: [{save_path}]!")
        is_crop = True
    else:
        logger.info(f"clip results have been exist! please check!")

    assert isinstance(cropsize, int)

    width, height, proj, geotrans = TifCrop(
        imgpath, save_path, cropsize, 0, logger, infer_id, is_crop
    )

    return save_path, width, height, proj, geotrans


class TqdmToLogger:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.pbar = None

    def write(self, msg):
        if self.pbar is None:
            self.logger.log(self.level, msg.rstrip())
        else:
            self.pbar.write(msg)

    def flush(self):
        pass


class DeployDataset(Dataset):
    def __init__(self, root: str):
        self.images_list = self._make_file_path_list(root)

    def __getitem__(self, index):
        image_path = self.images_list[index]

        return image_path

    def __len__(self):
        return len(self.images_list)

    def _make_full_path(self, root_list, root_path):
        file_full_path_list = []
        for filename in root_list:
            file_full_path = os.path.join(root_path, filename)
            file_full_path_list.append(file_full_path)

        return file_full_path_list

    def _make_file_path_list(self, image_root):
        if not os.path.exists(image_root):
            raise FileNotFoundError(
                f"dataset of cliped image save path:[{image_root}] does not exist!"
            )
        from natsort import natsorted

        image_list = natsorted(os.listdir(image_root))
        image_list = [img for img in image_list if img.endswith(".tif")]

        image_full_path_list = self._make_full_path(image_list, image_root)

        return image_full_path_list


def set_dataloader(
        root,
        batch_size: int = 32,
        num_workers: int = 0,
):
    dataset = DeployDataset(root=root)

    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return dataloader


def infer_process(
        model,
        dataloader,
        pred_save_path,
        im_geotrans,
        im_proj,
        pixel_threshold,
        logger: logging.Logger,
        infer_id,
):
    if not os.path.exists(pred_save_path):
        os.makedirs(pred_save_path)
    logger.info(f"model outputs save dir: [{pred_save_path}]!")

    batch_size = dataloader.batch_size

    model.eval()
    logger.info("------------------" * 3)
    logger.info("(start deploying)")
    with tqdm(
            total=len(dataloader), ncols=100, colour="#C0FF20", file=TqdmToLogger(logger)
    ) as pbar:
        for batch_index, imgs in enumerate(dataloader):
            logger.info(f"Processing item {batch_index}")
            # 执行一些操作
            outs = inference_model(model, imgs)
            for out_index, out in enumerate(outs):
                out = (
                    out.pred_sem_seg.data.squeeze(1)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
                out[out == 1] = 255

                _, count = np.unique(out, return_counts=True)
                if count[-1] <= pixel_threshold:
                    out = np.zeros((out.shape[-2], out.shape[-1]))

                save_path = os.path.join(
                    pred_save_path,
                    infer_id
                    + "_"
                    + str(batch_index * batch_size + out_index + 1)
                    + ".tif",
                )
                writeTiff(out, im_geotrans=im_geotrans, im_proj=im_proj, path=save_path)
            pbar.update(1)

        return


def stitchtif(
        ori_img_path,
        croped_path,
        output_path,
        output_name,
        size,
        logger: logging.Logger,
        infer_id,
):
    if not os.path.exists(ori_img_path):
        raise FileNotFoundError(f"ori_img_path: {croped_path} does not exist!")

    if not os.path.exists(croped_path):
        raise FileNotFoundError(f"croped_path: {croped_path} does not exist!")

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        logger.info(f"Infer results save dir: [{output_path}]!")

    output_name = output_name + ".tif"

    stitchTiff(
        ori_img_path,
        croped_path,
        output_path,
        output_name,
        size,
        repetition=0,
        logger=logger,
        infer_id=infer_id,
    )


def set_infermodel(
        config,
        checkpoint,
        device,
):
    model = init_model(config, checkpoint, device=device)

    logging.warning(f"Model weights loaded!")

    return model


def set_logger(
        level,
        logging_save_dir,
        infer_id,
):
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(logging_save_dir)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(level)
    logger.addHandler(console)

    logger.info(f"Starting inferring {infer_id}")
    return logger


def infer_fn(
        root_org,
        root_crop,
        root_pred,
        root_result,
        output_name,
        model,
        batch_size,
        num_workers,
        logger: logging.Logger,
        size,
        infer_id,
):
    clip_save_path, _, _, proj, geotrans = croptif(
        root_org,
        root_crop,
        cropsize=size,
        logger=logger,
        infer_id=infer_id,
    )

    dataloader = set_dataloader(
        root=clip_save_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    infer_process(
        model=model,
        pred_save_path=root_pred,
        dataloader=dataloader,
        im_geotrans=geotrans,
        im_proj=proj,
        pixel_threshold=0,
        logger=logger,
        infer_id=infer_id,
    )

    stitchtif(
        ori_img_path=root_org,
        croped_path=root_pred,
        output_path=root_result,
        output_name=output_name,
        size=size,
        logger=logger,
        infer_id=infer_id,
    )


def get_argparser():
    # 需要推理的影像名称
    infer_id = "hongshuliang_70_24_8"

    parser = argparse.ArgumentParser()

    # 生成结果的总路径（工作空间）
    parser.add_argument(
        "--workspace",
        type=str,
        default="work_dir",
        help="base dir of workspace",
    )

    # 当前任务的位移ID，直接选择的是影像名称
    parser.add_argument(
        "--infer_id",
        type=str,
        default=infer_id,
        help="infer_id",
    )

    # mmseg模型的config文件路径
    parser.add_argument(
        "--model_config",
        type=str,
        default="work_dir/model.py",
    )

    # mmseg模型权重文件路径
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="work_dir/model.pth",
    )

    # 推理设备，默认cuda:0
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        choices=["cuda:0", "cpu"],
        help="framework for segmentation recognition.",
    )

    # 推理影像的路径
    parser.add_argument(
        "--infer_primal_image_path",
        type=str,
        default=f"E:/{infer_id}.tif",
    )

    # 输出结果的名称，建议默认影像名称
    parser.add_argument("--output_name", type=str, default=f"{infer_id}")

    # 推理的batchsize大小
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch_size",
    )

    # 在进行裁剪时的大小
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="size of clip",
    )

    # 推理的num_workers，默认为0
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="num_workers",
    )

    # 是否删除中间文件(裁剪小图像、小图像的推理结果)，只保留最终的提取结果
    parser.add_argument(
        "--delete_Intermediate_products",
        type=bool,
        default=True,
        help="delete Intermediate products",
    )
    return parser


def infer():
    try:
        args = get_argparser().parse_args()

        if not os.path.exists(args.model_config):
            raise FileExistsError(f"{args.model_config} not exists!")
        if not os.path.exists(args.checkpoint):
            raise FileExistsError(f"{args.checkpoint} not exists!")

        model = set_infermodel(
            config=args.model_config,
            checkpoint=args.checkpoint,
            device=args.device,
        )

        INFER_CROP_SAVE_PATH = os.path.join(args.workspace, args.infer_id, "crop")
        INFER_PRED_SAVE_PATH = os.path.join(args.workspace, args.infer_id, "pred")
        INFER_RESULT_SAVE_PATH = os.path.join(args.workspace, args.infer_id, "result")
        logging_save_dir = os.path.join(args.workspace, args.infer_id, "log", "logging")
        logger = set_logger(logging.DEBUG, logging_save_dir, args.infer_id)
        infer_fn(
            root_org=args.infer_primal_image_path,
            root_crop=INFER_CROP_SAVE_PATH,
            root_pred=INFER_PRED_SAVE_PATH,
            root_result=INFER_RESULT_SAVE_PATH,
            output_name=args.output_name,
            model=model,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            size=args.size,
            logger=logger,
            infer_id=args.infer_id,
        )
        if args.delete_Intermediate_products:
            delete_dir(INFER_CROP_SAVE_PATH)
            logger.warning(f"The cropped image is clear!")

            delete_dir(INFER_PRED_SAVE_PATH)
            logger.warning(f"The predicted small image has been clear!")

        INFER_URL_SAVE_PATH = os.path.join(args.workspace, "infer", "org")
        if os.path.exists(INFER_URL_SAVE_PATH):
            delete_dir(INFER_URL_SAVE_PATH)
            logger.warning(f"The url image has been clear!")

        logging.info(f"infer work: {args.infer_id} has been finished!")

        return

    except Exception as e:
        # 捕获推理异常，记录错误信息
        error_message = str(e)
        logging.error(f"Infer Error:{error_message}")
        # 设置模型推理状态为异常失败
        # 返回错误信息给前端
        return


if __name__ == "__main__":
    infer()
