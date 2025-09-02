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
import itertools
import math
from hashlib import md5
import shutil
import gzip

# 3rd party
import matplotlib.pyplot as plt
from PIL import Image
import skimage as ski
import numpy as np
import torch
from tqdm.auto import tqdm

# DiDip
import fargv

# local
src_root = Path(__file__).parents[1]
sys.path.append( str( src_root ))
from bin import ddp_lineseg as lsg
from libs import seglib, list_utils as lu


logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(levelname)s: %(funcName)s - %(message)s", force=True )
logger = logging.getLogger(__name__)
# tone down unwanted logging
logging.getLogger('matplotlib.font_manager').disabled=True
logging.getLogger('PIL').setLevel(logging.INFO)


p = {
    'model_path': str(src_root.joinpath("best.mlmodel")),
    'mask_threshold': [0.25, "Threshold used for line masks--a tweak on the post-processing phase."],
    'box_threshold': [0.8, "Threshold used for line bounding boxes."],
    'rescale': [0, "If True, display segmentation on original image; otherwise (default), get the image size from the model used for inference (ex. 1024 x 1024)."],
    'img_paths': set([]),
    'limit': [0, "How many files to display."],
    'random': [0, "If non-null, randomly pick <random> paths out of the <img_paths> list."],
    'segfile_suffix': ['', "If a line segmentation suffix is provided (ex. 'lines.pred.json'), predicted lines are read from <img_path>.<suffix>."],
    'segfile': ['', "If a line segmentation file is provided, predicted lines are read from this file."],
    'patch_row_count': [ 0, "Process the image in <patch_row_count> rows."],
    'patch_col_count': [ 0, "Process the image in <patch_col_count> cols."],
    'patch_size': [0, "Process the image by <patch_size>*<patch_size> patches"],
    'icdar_threshold': 0.75,
    'foreground_only': [0, "Evaluate on foreground pixels only."],
    'output_file_name': ['',"Output file name; if prefixed with '>>', append to an existing file."],
    'save_file_scores': [1, "Save the detailed, per-file scores."],
    'cache_predictions': [1, "Cache prediction tensors for faster, repeated calls with various post-processing optiosn."],
    'output_root_dir': ['/tmp', "Where to save the cached predictions."],
    'method': [ ('icdar2017', 'iou'), "Evaluation method: 'icdar2017' checks both prec. and rec. separately for find TPs; 'iou' checks best IoU."],
}



def binary_map_from_patches( img: Image.Image, row_count=2, col_count=1, overlap=100, model=None, mask_threshold=.25, box_threshold=.8):
    """
    Construct a single label map from predictions on <row_count>x<col_count> patches.

    Args:
        img (Image.Image): a PIL image.
        row_count (int): number of rows.
        col_count (int): number of cols.
        overlap (int): overlap between patches (in pixels)
        mask_threshold (float): confidence score threshold for soft line masks.
        box_threshold (float): confidence score threshold for line bounding boxes.

    Returns:
        np.ndarray: a (1,H,W) binary map.
    """
    assert model is not None
    logger.debug("row_count={}, col_count={}".format(row_count, col_count))
    row_cuts_exact, col_cuts_exact  = [ list(int(f) for f in np.linspace(0, dim, d)) for dim, d in ((img.height, row_count+1), (img.width, col_count+1)) ]
    row_cuts, col_cuts = [[[ c+overlap, c-overlap] if c and c<cuts[-1] else c for c in cuts ] for cuts in ( row_cuts_exact, col_cuts_exact ) ]
    rows, cols = [ lu.group( lu.flatten( cut ), gs=2) for cut in (row_cuts, col_cuts) ]
    crops_yyxx=[ lu.flatten(lst) for lst in itertools.product( rows, cols ) ]
    logger.debug(crops_yyxx)
    img_hwc = np.array( img )
    img_crops = [ torch.from_numpy(img_hwc[ crop[0]:crop[1], crop[2]:crop[3] ]).permute(2,0,1) for crop in crops_yyxx ]
    _, crop_preds, crop_sizes = lsg.predict( img_crops, live_model=model )
    page_mask = np.zeros((crops_yyxx[-1][1],crops_yyxx[-1][3]), dtype='bool')
    for i in range(len(crops_yyxx)):
        t,b,l,r = crops_yyxx[i]
        patch_mask = lsg.post_process( crop_preds[i], orig_size=crop_sizes[i], mask_threshold=mask_threshold, box_threshold=box_threshold )
        if patch_mask is None:
            continue
        page_mask[t:b, l:r] += patch_mask[0]
    return page_mask[None,:]


def binary_map_from_fixed_patches( img: Image.Image, patch_size=1024, overlap=100, model=None, mask_threshold=.25, box_threshold=.8, cached_prediction_prefix='', cached_prediction_path=Path('/tmp')) -> np.ndarray:
    """
    Construct a single label map from predictions on patches of size <patch_size> x <patch_size>.

    Args:
        img (Image.Image): a PIL image.
        patch_size (int): size of the square patch.
        overlap (int): minimum overlap between patches (in pixels)
        mask_threshold (float): confidence score threshold for soft line masks.
        box_threshold (float): confidence score threshold for line bounding boxes.
        cached_prediction_prefix (str): a MD5 string for this image, to indicate that a prediction pickle should be checked for.
        cached_prediction_path (Path): where to save the cached predictions.

    Returns:
        np.ndarray: a (1,H,W) binary map.
    """
    assert model is not None
    img_hwc = np.array( img )
    height, width = img_hwc.shape[:2]

    # ensure that image is at least <patch_size> high and wide
    new_height = img_hwc.shape[0] if img_hwc.shape[0] >= patch_size else patch_size
    new_width = img_hwc.shape[1] if img_hwc.shape[1] >= patch_size else patch_size
    rescaled = False
    if new_height != img_hwc.shape[0] or new_width != img_hwc.shape[1]:
        img_hwc = ski.transform.resize( img_hwc, (new_height, new_width ))
        rescaled = True
    # cut into tiles
    tile_tls = seglib.tile_img( (new_width, new_height), patch_size, constraint=overlap )
    # Safety valve :)
    if  len(tile_tls) > 25:
        return None

    crop_preds = None
    cached_prediction_file = cached_prediction_path.joinpath( '{}.pt.gz'.format( cached_prediction_prefix ))
    ignore_cached_file = True
    if cached_prediction_prefix and cached_prediction_file.exists():
        ignore_cached_file = False
        try:
            uzpf = gzip.GzipFile( cached_prediction_file, 'r')
            crop_preds = torch.load( uzpf, weights_only=False)
        except RuntimeError as e:
            logger.warning("Runtime error {}".format(e))
            ignore_cached_file = True 
    if ignore_cached_file:
        img_crops = [ torch.from_numpy(img_hwc[y:y+patch_size,x:x+patch_size]).permute(2,0,1) for (y,x) in tile_tls ]
        logger.debug([ c.shape for c in img_crops ])
        
        _, crop_preds, _ = lsg.predict( img_crops, live_model=model )
        if cached_prediction_prefix:
            zpf = gzip.GzipFile( cached_prediction_file, 'w')
            torch.save(crop_preds, zpf)

    page_mask = np.zeros((img_hwc.shape[0],img_hwc.shape[1]), dtype='bool')
    for i,(y,x) in enumerate(tile_tls):
        patch_mask = lsg.post_process( crop_preds[i], mask_threshold=mask_threshold, box_threshold=box_threshold )
        if patch_mask is None:
            continue
        page_mask[y:y+patch_size, x:x+patch_size] += patch_mask[0]
    # resize to orig. size, if needed
    if rescaled:
        page_mask = ski.transform.resize( page_mask, (height, width ))
    return page_mask[None,:]



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
                binary_mask = binary_map_from_fixed_patches( Image.open(img_path), patch_size=patch_size, model=live_model, mask_threshold=args.mask_threshold, box_threshold=args.box_threshold, cached_prediction_prefix=img_md5, cached_prediction_path=cache_subdir_path )
            # Style 2: Inference M x N squares
            else:
                patch_row_count = args.patch_row_count if args.patch_row_count else 1
                patch_col_count = args.patch_col_count if args.patch_col_count else 1
                logger.debug("Patches: {}x{}".format(patch_row_count, patch_col_count))
                binary_mask = binary_map_from_patches( Image.open(img_path), patch_row_count, patch_col_count, model=live_model, mask_threshold=args.mask_threshold, box_threshold=args.box_threshold )

            logger.debug("Inference time: {:.5f}s".format( time.time()-start))

        # Default: Page-wide inference
        else:
            if 'train_style' in live_model.hyper_parameters and live_model.hyper_parameters['train_style'] == 'patch':
                logger.warning('The model being loaded was trained on fixed-size patches: expect suboptimal results.')
            imgs_t, preds, sizes = lsg.predict( [img_path], live_model=live_model)
            logger.debug("Inference time: {:.5f}s".format( time.time()-start))
            if args.rescale:
                logger.debug("Rescale")
                binary_mask = lsg.post_process( preds[0], orig_size=sizes[0], mask_threshold=args.mask_threshold, box_threshold=args.box_threshold )
            else:
                logger.debug("Square")
                # TODO: label binary map
                binary_mask= lsg.post_process( preds[0], mask_threshold=args.mask_threshold , box_threshold=args.box_threshold)
        if binary_mask is None:
            logger.warning("No line mask found for {}: skipping item.".format( img_path ))
            continue
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
        #np.save('pm.npy', pixel_metrics)
        pms.append( pixel_metrics )
    # pms is a list of 6-tuples (Match-threshold, TP, FP, FN, Jaccard, F1)
    eval_method = seglib.polygon_pixel_metrics_to_line_based_scores_icdar_2017 if args.method=='icdar2017' else seglib.polygon_pixel_metrics_to_line_based_scores
    raw_tuples = [ eval_method( pm, threshold=args.icdar_threshold ) for pm in pms ]
    iou_tp_fp_fn_prec_rec_jaccard_f1_8n = np.stack( [ rt for rt in raw_tuples if not np.sum(np.isnan( rt )) ], axis=1)

    # individual file scores are not saved when aggregate output only on stdout
    if args.save_file_scores:
        file_scores = zip( [ str(f) for f in files], iou_tp_fp_fn_prec_rec_jaccard_f1_8n.transpose().tolist())
        file_scores_str = '\n'.join([ '{}\t{}'.format(filename, '\t'.join([ str(s) for s in scores])) for filename, scores in file_scores ])
        file_scores_filepath = Path(output_subdir_path, f'file_scores_{args.box_threshold}_{args.mask_threshold}.tsv')
        with open( file_scores_filepath, 'w') as of:
            of.write('Img_path\tIoU\tTP\tFP\tFN\tPrecision\tRecall\tJaccard\tF1\n')
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


