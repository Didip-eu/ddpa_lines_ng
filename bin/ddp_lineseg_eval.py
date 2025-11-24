#!/usr/bin/env python3

# nprenet@gmail.com
# 05.2025

"""
Line segmentation: evaluation script.

The script is for evaluating the final outcome of both segmentation stages:

1. Mask R-CNN stage (computing line masks and boxes)
2. Post-processing stage (construction of the flat page-wide map)

Works for both page-wide inference and patch-based inference, depending on the model used.
"""

# stdlib
from pathlib import Path
import time
import sys
import random
import logging
from hashlib import md5
import shutil
import gzip

# 3rd party
from PIL import Image
import skimage as ski
import numpy as np
from tqdm.auto import tqdm


# DiDip
import fargv

# local
src_root = Path(__file__).parents[1]
sys.path.append( str( src_root ))
from bin import ddp_lineseg_train as lsg
from libs import seglib, list_utils as lu, line_geometry as lgm 


logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(levelname)s: %(funcName)s - %(message)s", force=True )
logger = logging.getLogger(__name__)
# tone down unwanted logging
logging.getLogger('matplotlib.font_manager').disabled=True
logging.getLogger('PIL').setLevel(logging.INFO)


p = {
    'model_path': str(src_root.joinpath("best.mlmodel")),
    'mask_threshold': [0.25, "Threshold used for line masks--a tweak on the post-processing phase."],
    'box_threshold': [0.8, "Threshold used for line bounding boxes."],
    'rescale': [False, "If True, display segmentation on original image; otherwise (default), get the image size from the model used for inference (ex. 1024 x 1024)."],
    'img_paths': set([]),
    'limit': [0, "How many files to display."],
    'random': [0, "If non-null, randomly pick <random> paths out of the <img_paths> list."],
    'segfile_suffix': ['', "If a line segmentation suffix is provided (ex. 'lines.pred.json'), predicted lines are read from <img_path>.<suffix>."],
    'segfile': ['', "If a line segmentation file is provided, predicted lines are read from this file."],
    'patch_row_count': [ 0, "Process the image in <patch_row_count> rows."],
    'patch_col_count': [ 0, "Process the image in <patch_col_count> cols."],
    'patch_size': [0, "Process the image by <patch_size>*<patch_size> patches"],
    'icdar_threshold': 0.75,
    'foreground_only': [False, "Evaluate on foreground pixels only."],
    'output_file_name': ['',"Output file name; if prefixed with '>>', append to an existing file."],
    'save_file_scores': [True, "Save the detailed, per-file scores."],
    'file_scores_prefix': ['file_scores', "String to be prepended to the per-file scores file (the suffix is made of the box- and mask thresholds."],
    'cache_predictions': [True, "Cache prediction tensors for faster, repeated calls with various post-processing optiosn."],
    'output_root_dir': ['/tmp', "Where to save the cached predictions."],
    'method': [ ('icdar2017', 'iou'), "Evaluation method: 'icdar2017' checks both prec. and rec. separately for find TPs; 'iou' checks best IoU."],
}


if __name__ == '__main__':

    args, _ = fargv.fargv(p)
    logger.debug( args )

    # Using cached predictions: raw prediction tensors (as given by Mask-RCNN before preprocessing) are 
    # written (and later retrieved) in a different directory for each model used. Ex.
    # <args.output_root_dir>
    #     |__ 657434a3452b61ca05fcb1011aac166a.mlmodel
    #     |__ 657434a3452b61ca05fcb1011aac166a
    #     |      |__ <args.output_file_name.tsv>
    #     |      |__ cache
    #     |            |__ 3b1a7985522752da4a21bb5fa8ebe963.pt
    #     |            |__ 418098bc2cbc0c878430737a410e1554.pt
    #     |            |__ ...
    #     |__ 7e50e7f159611756e2fdb11c056130ab.mlmodel
    #     |__ 7e50e7f159611756e2fdb11c056130ab
    #     |      |__ *.tsv
    #     |      |__ cache
    #     |            |__ 3b1a7985522752da4a21bb5fa8ebe963.pt
    #     |            |__ 418098bc2cbc0c878430737a410e1554.pt
    #     |            |__ ...
    output_file_name = args.output_file_name
    if output_file_name and Path(args.output_file_name).match('*/*'):
        output_file_name = Path(args.output_file_name).name
        logger.warning('Parameter -output_file_name cannot be a path: removing path directory component.')

    output_root_path = Path(args.output_root_dir)
    output_subdir_path = None
    cache_subdir_path = None
    with open( args.model_path, 'rb') as mf:
        # computing MD5 of model file used
        model_md5 = md5( mf.read() ).hexdigest()
        # create output subdir for this model
        output_subdir_path = output_root_path.joinpath( model_md5 )
        print(output_subdir_path)
        output_subdir_path.mkdir( exist_ok=True )
        model_local_copy_path = output_subdir_path.with_suffix('.mlmodel')
        # copy model file into root folder, with MD5 identifier (make it easier to rerun eval loops later)
        if not model_local_copy_path.exists():
            shutil.copy2( args.model_path, model_local_copy_path )
        if args.cache_predictions:
            cache_subdir_path = output_subdir_path.joinpath('cached') 
            cache_subdir_path.mkdir( exist_ok=True )
            logger.info( 'Using cache subdirectory {}.'.format( cache_subdir_path ))

    live_model = lsg.SegModel.load( args.model_path ) if (not args.segfile_suffix and not args.segfile) else None

    files = []
    if args.random:
        files = random.sample([ Path(p) for p in args.img_paths ], args.random)
    else:
        files = [ Path(p) for p in ( list(args.img_paths)[:args.limit] if args.limit else args.img_paths) ]

    # pixel metrics
    pms = []

    np.set_printoptions(linewidth=1000, edgeitems=10)
    for img_path in tqdm( files ):
        gt_map = seglib.gt_masks_to_labeled_map( seglib.line_binary_mask_stack_from_json_file( str(img_path).replace('img.jpg', 'lines.gt.json')))
        binary_mask, label_map_hw = None, None

        img_md5=''
        if args.cache_predictions:
            with open( img_path, 'rb') as imgf:
                img_md5 = md5( imgf.read()).hexdigest()
                #logger.info(f'{img_md5}')

        if live_model is None:
            sys.exit()

        start = time.time()

        # Step 1: stacks of binary masks (one for each detected boxes)
        if args.patch_size or args.patch_row_count or args.patch_col_count:

            # Style 1: inference from fixed-size squares
            if args.patch_size:
                patch_size = args.patch_size
                if 'train_style' in live_model.hyper_parameters:
                    if live_model.hyper_parameters['train_style'] != 'patch':
                        logger.warning('The model being loaded was _not_ trained on fixed-size patches: expect suboptimal results.')
                    elif live_model.hyper_parameters['img_size'][0] != args.patch_size:
                        logger.warning('The model being loaded is trained on {}x{} patches, but the script uses a {} patch size argument: overriding patch_size value with model-stored size.'.format( *live_model.hyper_parameters['img_size'], args.patch_size))
                        patch_size = live_model.hyper_parameters['img_size'][0]
                logger.debug('Patch size: {} x {}'.format( patch_size, patch_size))
                binary_mask = lgm.binary_mask_from_fixed_patches( Image.open(img_path), patch_size=patch_size, model=live_model, mask_threshold=args.mask_threshold, box_threshold=args.box_threshold, cached_prediction_prefix=img_md5, cached_prediction_path=cache_subdir_path )
            # Style 2: Inference M x N squares
            else:
                patch_row_count = args.patch_row_count if args.patch_row_count else 1
                patch_col_count = args.patch_col_count if args.patch_col_count else 1
                logger.debug("Patches: {}x{}".format(patch_row_count, patch_col_count))
                binary_mask = lgm.binary_mask_from_patches( Image.open(img_path), patch_row_count, patch_col_count, model=live_model, mask_threshold=args.mask_threshold, box_threshold=args.box_threshold )

            logger.debug("Inference time: {:.5f}s".format( time.time()-start))

        else:
            # Default: Page-wide inference
            if 'train_style' in live_model.hyper_parameters and live_model.hyper_parameters['train_style'] == 'patch':
                logger.warning('The model being loaded was trained on fixed-size patches: expect suboptimal results.')
            imgs_t, preds, sizes = lsg.predict( [img_path], live_model=live_model)
            logger.debug("Inference time: {:.5f}s".format( time.time()-start))
            if args.rescale:
                logger.debug("Rescale")
                binary_mask = lgm.post_process( preds[0], orig_size=sizes[0], mask_threshold=args.mask_threshold, box_threshold=args.box_threshold )
            else:
                logger.debug("Square")
                # TODO: label binary map
                binary_mask = lgm.post_process( preds[0], mask_threshold=args.mask_threshold , box_threshold=args.box_threshold)
        if binary_mask is None:
            logger.warning("No line mask found for {}: skipping item.".format( img_path ))
            continue

        # Step 2: flat map of integer labels
        label_map_hw = ski.measure.label( binary_mask, connectivity=2 ).squeeze()
        logger.debug("label_map.shape={}, label_map._dtype={}, max_label={}".format(label_map_hw.shape, label_map_hw.dtype, np.max(label_map_hw)))
        logger.debug("gt_map.shape={}, max_label={}".format(gt_map.shape, np.max(gt_map)))

        logger.debug('GT labels: {}, Pred labels: {}'.format( np.unique( gt_map ), np.unique( label_map_hw )))
        pixel_metrics=None
        if args.foreground_only:
            img_fg_mask = np.load('dataset/binary/{}'.format( img_path.name.replace('.img.jpg','.bin.npy')))
            pixel_metrics = seglib.polygon_pixel_metrics_two_flat_maps_and_mask( label_map_hw, gt_map, img_fg_mask ) 
        else:
            pixel_metrics = seglib.polygon_pixel_metrics_two_flat_maps( label_map_hw, gt_map ) 
            
        # pms is a list of 6-tuples (Match-threshold, TP, FP, FN, Jaccard, F1)
        pms.append( pixel_metrics )

    # Aggregating results from all img files
    eval_method = seglib.polygon_pixel_metrics_to_line_based_scores_icdar_2017 if args.method=='icdar2017' else seglib.polygon_pixel_metrics_to_line_based_scores
    raw_tuples = [ eval_method( pm, threshold=args.icdar_threshold ) for pm in pms ]
    iou_tp_fp_fn_prec_rec_jaccard_f1_8n = np.stack( [ rt for rt in raw_tuples if not np.sum(np.isnan( rt )) ], axis=1)


    if args.save_file_scores:
        # insert box and mask threshold columns 
        arr = iou_tp_fp_fn_prec_rec_jaccard_f1_8n
        arr = np.concatenate(( arr[:1], np.full((2,arr.shape[1]), [[args.box_threshold],[args.mask_threshold]] ), arr[1:]))
        file_scores = zip( [ str(f) for f in files], arr.transpose().tolist())
        file_scores_str = '\n'.join([ '{}\t{}'.format(filename, '\t'.join([ str(s) for s in scores])) for filename, scores in file_scores ])
        file_scores_filepath = Path(output_subdir_path, f'{args.file_scores_prefix}_{args.box_threshold}_{args.mask_threshold}.tsv')
        logger.info("Saving file scores into {}".format(file_scores_filepath))
        with open( file_scores_filepath, 'w') as of:
            of.write('Img_path\tIoU\tB-Thr\tM-Thr\tTP\tFP\tFN\tPrecision\tRecall\tJaccard\tF1\n')
            of.write( file_scores_str + '\n')

    # aggregate scores
    tps, fps, fns = np.sum( iou_tp_fp_fn_prec_rec_jaccard_f1_8n[1:4], axis=1)
    precs = tps / (tps+fps)
    recs = tps / (tps+fns)
    jaccard = tps / (tps+fps+fns) 
    f1 = 2*tps / ( 2*tps+fps+fns)

    if output_file_name:
        output_file = output_file_name.replace('>>','')
        append = output_file != output_file_name
        output_file_path = output_subdir_path.joinpath( output_file )
        with open( output_file_path, 'a' if append else 'w') as of:
            logger.info('Evaluation metrics saved on {}'.format( output_file_path ))
            if of.tell() == 0: # add header on empty file 
                of.write('IoU\tB-Thr\tM-Thr\tTP\tFP\tFN\tPrecision\tRecall\tJaccard\tF1\n')
            of.write(f'{args.icdar_threshold}\t{args.box_threshold}\t{args.mask_threshold}\t{tps}\t{fps}\t{fns}\t{precs}\t{recs}\t{jaccard}\t{f1}\n')
    else:
        print('IoU\tB-Thr\tM-Thr\tTP\tFP\tFN\tPrecision\tRecall\tJaccard\tF1')
        print(f'{args.icdar_threshold}\t{args.box_threshold}\t{args.mask_threshold}\t{tps}\t{fps}\t{fns}\t{precs}\t{recs}\t{jaccard}\t{f1}')


