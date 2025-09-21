"""
line_build.py

This module factors out functions that call the low-level line prediction routines on libs/ddp_lineseg_train.py 
and process their outcomes, made of boxes and a matching stack of soft masks.

Users:

+ `bin/ddp_lineseg_viewer.py`
+ `bin/ddp_lineseg_eval.py`
+ `bin/ddp_line_detect.py`

"""
import logging
from pathlib import Path
import gzip

from PIL import Image
import skimage as ski
import numpy as np
import torch

from bin import ddp_lineseg_train as lsg
from libs import seglib


logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(levelname)s: %(funcName)s - %(message)s", force=True )
logger = logging.getLogger(__name__)



def post_process( preds: dict, box_threshold=.75, mask_threshold=.6, orig_size=()):
    """
    Compute lines from predictions, by merging box masks.

    Args:
        preds (dict[str,torch.Tensor]): predicted dictionary for the page:
            - 'scores'(N) : box probs
            - 'masks' (N1HW): line heatmaps
            - 'orig_size': if provided, masks are rescaled to the respective size
    Returns:
         np.ndarray: binary mask (1,H,W)
    """
    # select masks with best box scores
    best_masks = [ m.detach().numpy() for m in preds['masks'][preds['scores']>=box_threshold]]
    if len(best_masks) < preds['masks'].shape[0]:
        logger.debug("Selecting masks {} out of {}".format( np.argwhere( preds['scores']>=box_threshold ).tolist(), len(preds['scores'])))
    if not best_masks:
        return None
    # threshold masks
    masks = [ m * (m >= mask_threshold) for m in best_masks ]
    # merge masks 
    page_wide_mask_1hw = np.sum( masks, axis=0 ).astype('bool')
    # optional: scale up masks to the original size of the image
    if orig_size:
        page_wide_mask_1hw = ski.transform.resize( page_wide_mask_1hw, (1, orig_size[1], orig_size[0]))
    return page_wide_mask_1hw


def post_process_boxes( preds: dict, box_threshold=.9, mask_threshold=.1, orig_size=()):
    """
    Compute lines from predictions, by separate processing of box masks.
    (UNUSED)

    Args:
        preds (dict[str,torch.Tensor]): predicted dictionary for the page:
            - 'scores'(N) : box probs
            - 'masks' (N1HW): line heatmaps
            - 'orig_size': if provided, masks are rescaled to the respective size
    Returns:
        tuple[ np.ndarray, list[tuple[int, list, float, list]]]: a pair with
            - binary mask (1,H,W)
            - a list of line attribute dicts (label, centroid pt, area, polygon coords, ...)
    """
    # select masks with best box scores
    best_masks = [ m.detach().numpy() for m in preds['masks'][preds['scores']>box_threshold]]
    # threshold masks
    masks = [ (m * (m > mask_threshold)).astype('bool') for m in best_masks ]
    # in each mask, keep the largest CC
    clean_masks = []
    for m_1hw in masks:
        labeled_msk_1hw = ski.measure.label( m_1hw, connectivity=2 )
        reg_props = ski.measure.regionprops( labeled_msk_1hw )
        max_label, _ = max([ (reg.label, reg.area) for reg in reg_props ], key=lambda t: t[1])
        clean_masks.append( m_1hw * (labeled_msk_1hw == max_label))

    # merge masks 
    page_wide_mask_1hw = np.sum( clean_masks, axis=0 ).astype('bool')
    plt.imshow( np.squeeze(np.sum( clean_masks, axis=0)) )
    plt.show()
    # optional: scale up masks to the original size of the image
    if orig_size:
        page_wide_mask_1hw = ski.transform.resize( page_wide_mask_1hw, (1, orig_size[1], orig_size[0]))

    return page_wide_mask_1w


def get_morphology( page_wide_mask_1hw: np.ndarray, centerlines=False, polygon_area_threshold=100):
    """
    From a page-wide line mask, extract a labeled map and a dictionary of features.
    
    Args:
        page_wide_mask_1hw (np.ndarray): a binary line mask (1,H,W)
        centerlines (bool): compute the centerline and average height for each line.
        polygon_area_threshold (int): minimum number of pixels for a polygon to survive.
    Returns:
        tuple[ np.ndarray, list[tuple[int, list, float, list]]]: a pair with
            - labeled map(H,W)
            - a list of line attribute dicts (label, centroid pt, area, polygon coords, ...)
    """
    
    # label components
    labeled_msk_hw = ski.measure.label( page_wide_mask_1hw[0], connectivity=2 )
    # remove components that do not pass threshold
    for lbl in range(1, np.max(labeled_msk_hw) + 1):
        if np.sum( labeled_msk_hw == lbl ) < polygon_area_threshold:
            labeled_msk_hw *= ~(labeled_msk_hw==lbl)
    labels = np.unique( labeled_msk_hw[ labeled_msk_hw > 0 ] )

    logger.debug("Found {} connected components on 1HW binary map.".format( len(labels)))
    # sort label from top to bottom (using centroids of labeled regions) 
    line_region_properties = ski.measure.regionprops( labeled_msk_hw )
    logger.debug("Extracted {} region property records from labeled map.".format( len( line_region_properties )))
    # list of line attribute tuples 
    attributes = sorted([ (reg.label, reg.centroid, reg.area ) for reg in line_region_properties ], key=lambda attributes: (attributes[1][0], attributes[1][1]))
    
    polygon_coords = []
    line_heights = [-1] * len(labels)
    skeleton_coords = [ [] for i in labels ]

    for lbl in labels:
        lbl_count = np.sum( labeled_msk_hw == lbl )
        polygon_coords.append( ski.measure.find_contours( labeled_msk_hw == lbl )[0].astype('int'))

    if centerlines:
        binary_msk_hw = ( labeled_msk_hw > 0  ).astype('int')
        page_wide_skeleton_hw = ski.morphology.skeletonize( binary_msk_hw )
        _ , distance = ski.morphology.medial_axis( page_wide_skeleton_hw, return_distance=True )
        labeled_skl = ski.measure.label( page_wide_skeleton_hw, connectivity=2)
        logger.debug("Computed {} skeletons on 1HW binary map.".format( np.max( labeled_skl )))
        skeleton_coords = [ reg.coords for reg in ski.measure.regionprops( labeled_skl ) ]
        line_heights = []
        logger.debug("Computing line heights...")
        for lbl in range(1, np.max(labeled_skl)+1):
            line_skeleton_dist = page_wide_skeleton_hw * ( labeled_skl == lbl ) * distance 
            logger.debug("- labeled skeleton {} of length {}".format(lbl, np.sum(line_skeleton_dist != 0) ))
            line_heights.append( (np.mean(line_skeleton_dist[ line_skeleton_dist != 0])*2).item() )
        assert len(line_heights) == len( line_region_properties ) 

    # BBs centroid ordering differs from CCs top-to-bottom ordering:
    # usually hints at messy, non-standard line layout
    if [ att[0] for att in attributes ] != sorted([att[0] for att in attributes]):
        logger.warning("Labels may not follow reading order.")

    entry = (labeled_msk_hw[None,:], [{
                'label': att[0], 
                'centroid': att[1], 
                'area': att[2], 
                'polygon_coords': plgc,
                'line_height': lh, 
                'centerline': skc,
            } for att, lh, skc, plgc in zip(attributes, line_heights, skeleton_coords, polygon_coords) ])
    return entry


def binary_mask_from_patches( img: Image.Image, row_count=2, col_count=1, overlap=100, model=None, mask_threshold=.25, box_threshold=.8):
    """
    Construct a single binary mask from predictions on <row_count>x<col_count> patches.

    Args:
        img (Image.Image): a PIL image.
        row_count (int): number of rows.
        col_count (int): number of cols.
        overlap (int): overlap between patches (in pixels)
        mask_threshold (float): confidence score threshold for soft line masks.
        box_threshold (float): confidence score threshold for line bounding boxes.

    Returns:
        np.ndarray: a (1,H,W) binary_mask.
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
        patch_mask = post_process( crop_preds[i], orig_size=crop_sizes[i], mask_threshold=mask_threshold, box_threshold=box_threshold )
        if patch_mask is None:
            continue
        page_mask[t:b, l:r] += patch_mask[0]
    return page_mask[None,:]


def binary_mask_from_fixed_patches( img: Image.Image, patch_size=1024, overlap=100, model=None, mask_threshold=.25, box_threshold=.8, cached_prediction_prefix='', cached_prediction_path=Path('/tmp'), max_patches=25) -> np.ndarray:
    """
    Construct a single binary mask from predictions on patches of size <patch_size> x <patch_size>.

    Args:
        img (Image.Image): a PIL image.
        patch_size (int): size of the square patch.
        overlap (int): minimum overlap between patches (in pixels)
        mask_threshold (float): confidence score threshold for soft line masks.
        box_threshold (float): confidence score threshold for line bounding boxes.
        cached_prediction_prefix (str): a MD5 string for this image, to indicate that a prediction pickle should be checked for.
        cached_prediction_path (Path): where to save the cached predictions.

    Returns:
        np.ndarray: a (1,H,W) binary mask.
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
    if  len(tile_tls) > max_patches:
        logger.warning("Image slices into {} 1024-pixel patches: limit ({}) exceeded.".format(len(tile_tls), max_patches))
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
        patch_mask = post_process( crop_preds[i], mask_threshold=mask_threshold, box_threshold=box_threshold )
        if patch_mask is None:
            continue
        page_mask[y:y+patch_size, x:x+patch_size] += patch_mask[0]
    # resize to orig. size, if needed
    if rescaled:
        page_mask = ski.transform.resize( page_mask, (height, width ))
    return page_mask[None,:]



