#!/usr/bin/env python3

# nprenet@gmail.com
# 05.2025

"""
A simple line segmenter/viewer that predicts lines on images and displays the result.

A segmenter just for the eyes, or a viewer that can segment if needed: it stands by itself, with no regard for export functionalities.
For proper segmentation and recording of a region-based segmentation (crops), see `ddp_line_detect.py`, that is meant
to included into an HTR pipeline.

Examples:

    With image used as-is (no layout analysis), on-the-fly prediction and display:

    ```
    PYTHONPATH=. bin/ddp_lineseg_viewer.py -random 10 -model_path ./models/best_101_1024_bsz4.mlmodel -rescale 1 -img_paths ./dataset/*.jpg 
    ```

    Display an existing segmentation:

    ```
    PYTHONPATH=. bin/ddp_lineseg_viewer.py -random 10 -segfile_suffix lines.pred.json  -img_paths ./dataset/*.jpg
    ```

    High-quality segmentation of a COUS (Charter of Unusual Size), running inference separately on 3x1 patches:

    ```
    PYTHONPATH=. bin/ddp_lineseg_viewer.py -img_paths data/hard_cases/591e0762397178ee89e4c8b356be0da3.Wr_OldText.3.img.jpg -model_path ./models/best_101_1024_bsz4.mlmodel -patch_row_count 3
    ```

    Assuming the model has been trained on fixed-size crops of the charter image, inference can be run accordingly:

    ```
    PYTHONPATH=. bin/ddp_lineseg_viewer.py -img_paths data/hard_cases/591e0762397178ee89e4c8b356be0da3.Wr_OldText.3.img.jpg -model_path ./models/best_patch_1024.mlmodel -patch_size 1024
    ```

For proper segmentation and recording of a region-based segmentation (crops), see `ddp_line_detect.py`.'
"""

# stdlib
from pathlib import Path
import time
import sys
import random
import logging

# 3rd party
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# DiDip
import fargv

# local
src_root = Path(__file__).parents[1]
sys.path.append( str( src_root ))
from bin import ddp_lineseg_train as lsg
from libs import segviz, seglib, list_utils as lu, line_build as lb




logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(levelname)s: %(funcName)s - %(message)s", force=True )
logger = logging.getLogger(__name__)
# tone down unwanted logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)#.disabled=True
logging.getLogger('PIL').setLevel(logging.INFO)


p = {
    'model_path': str(src_root.joinpath("best.mlmodel")),
    'box_threshold': [0.75, "Threshold used for line bounding boxes."],
    'mask_threshold': [0.6, "Threshold used for line masks--a tweak on the post-processing phase."],
    'rescale': [0, "If True, display segmentation on original image; otherwise (default), get the image size from the model used for inference (ex. 1024 x 1024)."],
    'img_paths': set([]), #set(Path('dataset').glob('*.jpg')),
    'color_count': [0, "Number of colors for polygon overlay: -1 for single color, n > 1 for fixed number of colors, 0 for 1 color/line."],
    'limit': [0, "How many files to display."],
    'random': [0, "If non-null, randomly pick <random> paths out of the <img_paths> list."],
    'segfile_suffix': ['', "If a line segmentation suffix is provided (ex. 'lines.pred.json'), predicted lines are read from <img_path>.<suffix>."],
    'segfile': ['', "If a line segmentation file is provided, predicted lines are read from this file."],
    'patch_row_count': [ 0, "Process the image in <patch_row_count> rows."],
    'patch_col_count': [ 0, "Process the image in <patch_col_count> cols."],
    'patch_size': [0, "Process the image by <patch_size>*<patch_size> patches"],
    'show': set(['polygons', 'centerlines', 'regions', 'labels', 'title']),
    'linewidth': 2,
    'output_file_path': ['', 'If path is a directory, save the plot in <output_file_path>/<img_name_stem>.png.; otherwise, save under the provided file path.'],
    'crop_x': [1.0, "crop-in ratio (x axis, centered)"],
    'crop_y': [1.0, "crop-in ratio (y axis, centered)"],

}



if __name__ == '__main__':

    args, _ = fargv.fargv(p)
    logger.debug( args )

    live_model = lsg.SegModel.load( args.model_path ) if (not args.segfile_suffix and not args.segfile) else None

    files = []
    if args.random:
        files = random.sample([ Path(p) for p in args.img_paths ], args.random)
    else:
        files = [ Path(p) for p in ( list(args.img_paths)[:args.limit] if args.limit else args.img_paths) ]

    for img_path in files:
        logger.info(img_path)

        if live_model:
            start = time.time()
            mp, atts, path = None, None, None
            start = time.time()

            if args.patch_size or args.patch_row_count or args.patch_col_count:

                # Style 1: Inference fixed-size squares
                if args.patch_size:
                    patch_size = args.patch_size
                    if 'train_style' in live_model.hyper_parameters:
                        if live_model.hyper_parameters['train_style'] != 'patch':
                            logger.warning('The model being loaded was _not_ trained on fixed-size patches: expect suboptimal results.')
                        elif live_model.hyper_parameters['img_size'][0] != args.patch_size:
                            logger.warning('The model being loaded is trained on {}x{} patches, but the script uses a {} patch size argument: overriding patch_size value with model-stored size.'.format( *live_model.hyper_parameters['img_size'], args.patch_size))
                            patch_size = live_model.hyper_parameters['img_size'][0]
                    logger.debug('Patch size: {} x {}'.format( patch_size, patch_size))
                    binary_mask = lb.binary_mask_from_fixed_patches( Image.open(img_path), patch_size=patch_size, model=live_model, box_threshold=args.box_threshold, mask_threshold=args.mask_threshold )
                # Style 2: Inference M x N squares
                else:
                    patch_row_count = args.patch_row_count if args.patch_row_count else 1
                    patch_col_count = args.patch_col_count if args.patch_col_count else 1
                    logger.debug("Patches: {}x{}".format(patch_row_count, patch_col_count))
                    binary_mask = lb.binary_mask_from_patches( Image.open(img_path), patch_row_count, patch_col_count, model=live_model, box_threshold=args.box_threshold, mask_threshold=args.mask_threshold )
                if binary_mask is None:
                    logger.info("Invalid mask: skipping img {}".format( img_path))
                    continue

                logger.debug("Inference time: {:.5f}s".format( time.time()-start))
                logger.debug("binary_mask.shape={}".format(binary_mask.shape))
                segmentation_record = lb.get_morphology( binary_mask )
                logger.debug("segmentation_record[0].shape={}".format(segmentation_record[0].shape))
                mp, atts, path = segviz.batch_label_maps_to_img( [img_path], [segmentation_record], color_count=0 )[0]

            # Default: Page-wide inference
            else:
                if 'train_style' in live_model.hyper_parameters and live_model.hyper_parameters['train_style'] == 'patch':
                    logger.warning('The model being loaded was trained on fixed-size patches: expect suboptimal results.')
                imgs_t, preds, sizes = lsg.predict( [img_path], live_model=live_model)
                logger.debug("Inference time: {:.5f}s".format( time.time()-start))
                if args.rescale:
                    logger.debug("Rescale")
                    binary_mask = lb.post_process( preds[0], orig_size=sizes[0], box_threshold=args.box_threshold, mask_threshold=args.mask_threshold )
                    if binary_mask is None:
                        logger.warning("No line mask found for {}: skipping.".format( img_path ))
                        continue
                    logger.debug("binary_mask.shape={}".format(binary_mask.shape))
                    segmentation_record = lb.get_morphology( binary_mask )
                    mp, atts, path = segviz.batch_label_maps_to_img( [img_path], [segmentation_record], color_count=0 )[0]
                else:
                    logger.debug("Square")
                    binary_mask = lb.post_process( preds[0], box_threshold=args.box_threshold, mask_threshold=args.mask_threshold )
                    if binary_mask is None:
                        logger.warning("No line mask found for {}: skipping.".format( img_path ))
                        continue
                    segmentation_records= lb.get_morphology( binary_mask )
                    mp, atts, path = segviz.batch_label_maps_to_img( [ {'img':imgs_t[0], 'id':str(img_path)} ], [segmentation_records], color_count=0 )[0]
            logger.debug("Rendering time: {:.5f}s".format( time.time()-start))

            height, width = mp.shape[:2]
            delta_x, delta_y = (1-args.crop_x)/2, (1-args.crop_y)/2
            plt.close()
            plt.subplots(figsize=(12,12), dpi=144)
            plt.imshow( mp )
            plt.xlim( delta_x , width-delta_x)
            plt.ylim( height-delta_y, delta_y)
            if 'title' in args.show:
                plt.title( path )
            for att_dict in atts:
                if 'labels' in args.show:
                    label, centroid = att_dict['label'], att_dict['centroid']
                    plt.text(*centroid[:0:-1], label, size=15)
                if 'centerlines' in args.show:
                    centerline = att_dict['centerline']
                    plt.plot(*(centerline.transpose()[::-1]))
                if 'baselines' in args.show:
                    baseline = att_dict['baseline']
                    plt.plot(*(baseline.transpose()[::-1]))
            if args.output_file_path:
                #plt.subplots_adjust(0.2,0.075,0.90,0.95,0,0)
                output_file_path = Path( args.output_file_path, img_path.stem).with_suffix('.png') if Path(args.output_file_path).is_dir() else Path(args.output_file_path)
                plt.savefig( output_file_path, bbox_inches='tight')
            else:
                plt.show()
        else:
            if args.segfile:
                segviz.display_segmentation_and_img( img_path, segfile=args.segfile, show={ k:True for k in args.show if k != 'labels'}, linewidth=args.linewidth, crop=(args.crop_x, args.crop_y), output_file_path=args.output_file_path )
            elif args.segfile_suffix:
                segviz.display_segmentation_and_img( img_path, segfile_suffix=args.segfile_suffix, show={ k:True for k in args.show if k != 'labels'}, linewidth=args.linewidth, crop=(args.crop_x, args.crop_y), output_file_path=args.output_file_path )

