# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmengine.model import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model, show_result_pyplot
import os
import tqdm
import glob


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--img',
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.8,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--with-labels',
        action='store_true',
        default=False,
        help='Whether to display the class labels.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    if len(args.img) == 1:
        args.img = glob.glob(os.path.expanduser(args.img[0]))
        assert args.img, "The input path(s) was not found"
    for path in tqdm.tqdm(args.img, disable=not args.out_file):
        out_file = os.path.join(args.out_file, path.split('/')[-1])
        result = inference_model(model, path)
        # show the results
        show_result_pyplot(
            model,
            path,
            result,
            title=args.title,
            opacity=args.opacity,
            with_labels=args.with_labels,
            draw_gt=False,
            show=False if args.out_file is not None else True,
            out_file=out_file,
        )


if __name__ == '__main__':
    main()
