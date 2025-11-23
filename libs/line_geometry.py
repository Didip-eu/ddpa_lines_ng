"""
line_geometry.py

This module factors out 

+ Functions that call the low-level line prediction routines on libs/ddp_lineseg_train.py and process their outcomes, made of boxes and a matching stack of soft masks.
+ More broadly, functions that handle conversions between pixels maps and polygons, or line-to-polygon and polygon-to-polygon transformations.

Users:

+ `bin/ddp_lineseg_viewer.py`
+ `bin/ddp_lineseg_eval.py`
+ `bin/ddp_line_detect.py`

"""
import logging
from pathlib import Path
import gzip
from time import time
import random
from typing import Union

from PIL import Image, ImageDraw
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
    best_masks = [ m.detach().numpy() for m in preds['masks'][preds['scores']>=box_threshold].cpu()]
    if len(best_masks) < preds['masks'].shape[0]:
        logger.debug("Selecting masks {} out of {}".format( np.argwhere( preds['scores'].cpu()>=box_threshold ).tolist(), len(preds['scores'])))
        #pass
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


def get_morphology( page_wide_mask_1hw: np.ndarray, polygon_area_threshold=100, contour_tolerance=4, raw_polygons=False, height_factor=1.0):
    """
    From a page-wide line mask, extract a labeled map and a dictionary of features.
    
    Args:
        page_wide_mask_1hw (np.ndarray): a binary line mask (1,H,W)
        polygon_area_threshold (int): minimum number of pixels for a polygon to survive.
        contour_tolerance (int): max. distance constraint for line/polygon approximations.
        raw_polygons (bool): if True, return the (approximated) polygon obtained from the page mask; otherwise (default),
            return a reconstructed version of the polygon (baseline+height).
        height_factor (float): factor (∈  ]0,1]) to be applied to the polygon height-unused if 'raw_polygons' set.

    Returns:
        tuple[ np.ndarray, list[tuple[int, list, float, list]]]: a pair with
            - labeled map (H,W), with a choice of 2 flavors:
              + raw: as obtained by merging the line masks computed by the network
              + export-ready: a regularized version, constructed from the morphological features of the raw maps;
                it matches the polygon that is ultimately exported in the dictionary.
            - a list of line attribute dicts (label, centroid pt, area, polygon coords, ...)
    """
    # label components
    labeled_msk_hw = ski.measure.label( page_wide_mask_1hw[0], connectivity=2 )
    # remove components that do not pass threshold
    for lbl in range(1, np.max(labeled_msk_hw) + 1):
        if np.sum( labeled_msk_hw == lbl ) < polygon_area_threshold:
            labeled_msk_hw *= ~(labeled_msk_hw==lbl)
    labels = np.unique( labeled_msk_hw[ labeled_msk_hw > 0 ] )
    logger.debug("Found {} connected components on 1HW binary map ({}).".format( len(labels), labels))

    polygon_coords = []
    skeleton_coords = [] # a list of ndarrays
    line_heights = [] # a list of integers
    centroids = []

    def fix_ends( skl_yx: np.array, line_height: int, box_width: tuple[int,int]):
        """
        After pruning, skeleton's very ends may deviate markedly from the main axis; what follows is a crude,
        but adequate fix: both ends are truncated (proper length determined from line height) and replaced by a single 
        point at same height of the left(right)most point of the truncated skeleton. Option
        """
        x_leftmost, x_rightmost = np.min(skl_yx[:,1]), np.max(skl_yx[:,1])
        end_segment_length = int(np.ceil(line_height/(2*np.tan(np.pi/6))))
        skl_yx_reduced = skl_yx[end_segment_length-1:-end_segment_length+1]
        skl_yx_reduced[[0,-1]]=[[skl_yx[end_segment_length,0], x_leftmost],[ skl_yx[-(end_segment_length+1), 0], x_rightmost]]
        if skl_yx_reduced[0,1]>0:
            #skl_yx_reduced = np.concat( [ [[skl_yx_reduced[0,0], 0]], skl_yx_reduced ])
            skl_yx_reduced[0,1] =  0
        if skl_yx_reduced[-1,1]<box_width-1:
            #skl_yx_reduced = np.concat( [ skl_yx_reduced, [[skl_yx_reduced[-1,0], box_width-1]]] )
            skl_yx_reduced[-1,1] = box_width-1
        return skl_yx_reduced

    labeled_msk_regular_hw = None if raw_polygons else np.zeros(labeled_msk_hw.shape, dtype=labeled_msk_hw.dtype)

    logger.debug("Label processing")
    time_start = time()
    for lbl in labels:
        label_start = time()
        boundaries_nyx = ski.measure.find_contours( labeled_msk_hw == lbl )[0].astype('int')
        # simplifying polygon
        approximate_coords = ski.measure.approximate_polygon( boundaries_nyx, tolerance=contour_tolerance )
        polygon_coords.append( approximate_coords if len(approximate_coords) > 10 else boundaries_nyx )

        # 1. Create box with simplified polygon
        min_y, min_x = np.min( polygon_coords[-1], axis=0)
        max_y, max_x = np.max( polygon_coords[-1], axis=0)
        coords = (polygon_coords[-1] - np.array([min_y, min_x]))

        # PIL polygon fill is faster than skimage (by an order of magnitude)
        #polygon_box_ski = ski.draw.polygon2mask((max_y-min_y+1, max_x-min_x+1), coords )
        polygon_box=polygon_to_mask_pil( (max_y-min_y+1, max_x-min_x+1), coords )
        
        # 2. Skeletonize and prune
        try:
            _, this_skeleton_yx = prune_skeleton( ski.morphology.skeletonize( polygon_box ))
            # 3. Avg line height = area of polygon / length of skeleton
            line_heights.append( (np.sum(polygon_box) // len( this_skeleton_yx)).item() )
            this_skeleton_yx = fix_ends( this_skeleton_yx, line_heights[-1], polygon_box.shape[1] )
            centroids.append( this_skeleton_yx[int(len(this_skeleton_yx)/2)] + np.array( [min_y, min_x] ))
            approximate_pagewide_skl_yx = ski.measure.approximate_polygon(this_skeleton_yx, tolerance=3) + np.array( [min_y, min_x] )
            skeleton_coords.append( approximate_pagewide_skl_yx )

            if not raw_polygons:
                polyg = strip_from_centerline( skeleton_coords[-1][:,::-1], line_heights[-1]*height_factor )[:,::-1]
                polyg = boxed_in( polyg, (0,0,*[ d-1 for d in labeled_msk_hw.shape] ))
                polygon_coords[-1] = polyg
                polyg_rr, polyg_cc = ski.draw.polygon( *(polygon_coords[-1]).transpose())
                labeled_msk_regular_hw[ polyg_rr, polyg_cc ]=lbl
        #except (ValueError, IndexError) as e:
        except Exception as e:
            logger.warning("Failed to retrieve geometry from label mask #{}: {}".format(lbl, e))
        logger.debug("Done processing label {} - time: {:.5f}".format( lbl, time()-label_start ))
    logger.debug("Total label processing time: {:.5f}".format( time() - time_start ))
        
    # sort by centroids (y,x): 
    # - a very naive reading order heuristic, that does not work on multi-component, skewed lines
    #   Eg. (les feuilles mortes) --> ! mortes feuilles Les
    #                         mortes
    #               feuilles
    #          Les 
    # - order that differs from labels may hint at messy reading order
    line_features = sorted( zip(labels, line_heights, skeleton_coords, polygon_coords, centroids), key=lambda t: t[4].tolist() )
    return (labeled_msk_hw[None,:] if raw_polygons else labeled_msk_regular_hw[None,:], [{
                'label': lbl,
                'centroid': center_yx,
                'polygon_coords': plgc,
                'line_height': lh, 
                'centerline': skc,
                'baseline': skc + [lh/2,0],
    } for lbl, lh, skc, plgc, center_yx in line_features ])



def binary_mask_from_patches( img: Image.Image, row_count=2, col_count=1, overlap=.04, model=None, mask_threshold=.25, box_threshold=.8):
    """
    Construct a single binary mask from predictions on <row_count>x<col_count> patches.

    Args:
        img (Image.Image): a PIL image.
        row_count (int): number of rows.
        col_count (int): number of cols.
        overlap (float): minimum overlap between patches (a ratio to be applied to the image's largest dimension).
        mask_threshold (float): confidence score threshold for soft line masks.
        box_threshold (float): confidence score threshold for line bounding boxes.

    Returns:
        np.ndarray: a (1,H,W) binary_mask.
    """
    assert model is not None
    logger.debug("row_count={}, col_count={}".format(row_count, col_count))
    row_cuts_exact, col_cuts_exact  = [ list(int(f) for f in np.linspace(0, dim, d)) for dim, d in ((img.height, row_count+1), (img.width, col_count+1)) ]
    overlap_pixels = int(overlap * max(img.height, img.width))
    row_cuts, col_cuts = [[[ c+overlap_pixels, c-overlap_pixels] if c and c<cuts[-1] else c for c in cuts ] for cuts in ( row_cuts_exact, col_cuts_exact ) ]
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


def binary_mask_from_fixed_patches( img: Image.Image, patch_size=1024, overlap=.04, model=None, mask_threshold=.25, box_threshold=.8, cached_prediction_prefix='', cached_prediction_path=Path('/tmp'), max_patches=16, device='cpu') -> np.ndarray:
    """
    Construct a single binary mask from predictions on patches of size <patch_size> x <patch_size>.

    Args:
        img (Image.Image): a PIL image.
        patch_size (int): size of the square patch.
        overlap (float): minimum overlap between patches (a ratio to be applied to the image's largest dimension).
        mask_threshold (float): confidence score threshold for soft line masks.
        box_threshold (float): confidence score threshold for line bounding boxes.
        cached_prediction_prefix (str): a MD5 string for this image, to indicate that a prediction pickle should be checked for; a pickled prediction stores a list with one dictionary for each image crop.
        cached_prediction_path (Path): where to save the cached predictions.
        device (str): computing device - 'cuda' or 'cpu' (default).

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
    # ( tile-cutting + resize) until manageable
    resize_factor = 1.5 
    while True:
        tile_tls = seglib.tile_img( (new_width, new_height), patch_size, constraint=int(overlap*max(width,height)) )

        # small, single-patch images with a high estimated line count (>25) enlarged for better result
        if len(tile_tls) == 1:
            est_lc = line_count_estimate( img ) 
            if est_lc < 25:
                break
            new_height, new_width = int(new_height*2), int(new_width*2)
            img_hwc = ski.transform.resize( img_hwc, (new_height, new_width))
            logger.debug("Enlarging single-patch image based on line count (estimate={}).".format( est_lc ))
            rescaled=True
            continue
        # typical case: reasonable number of patches, no resizing
        elif  len(tile_tls) <= max_patches:
            logger.debug("Sliced image: {} patches.".format( len(tile_tls)))
            break
        # large images need to be reduced
        logger.debug("Image slices into {} 1024-pixel patches: limit ({}) exceeded.".format(len(tile_tls), max_patches))
        new_height, new_width = int(new_height/resize_factor), int(new_width/resize_factor)
        img_hwc = ski.transform.resize( img_hwc, (new_height, new_width)) 
        logger.debug("Resizing to ({}, {})".format( *img_hwc.shape[1::-1] ))
        rescaled = True

    crop_preds = None
    cached_prediction_file = Path(cached_prediction_path).joinpath( '{}.pt.gz'.format( cached_prediction_prefix )) if (cached_prediction_path and cached_prediction_prefix) else None
    ignore_cached_file = True
    if cached_prediction_file is not None and cached_prediction_file.exists():
        ignore_cached_file = False
        try:
            uzpf = gzip.GzipFile( cached_prediction_file, 'r')
            crop_preds = torch.load( uzpf, weights_only=False)
        except RuntimeError as e:
            logger.warning("Runtime error {}".format(e))
            ignore_cached_file = True 
        if len(crop_preds) != len(tile_tls):
            logger.warning("The number of cached predictions and the number of tiles differ; this typically happens when text crops (as provided by the layout analyzer) or the tile size have changed: refreshing the cached file ({}) instead.".format(cached_prediction_file))
            ignore_cached_file = True
    if ignore_cached_file:
        logger.debug('Ignoring cached files.')
        img_crops = [ torch.from_numpy(img_hwc[y:y+patch_size,x:x+patch_size]).permute(2,0,1) for (y,x) in tile_tls ]
        logger.debug([ c.shape for c in img_crops ])
        
        _, crop_preds, _ = lsg.predict( img_crops, live_model=model, device=device )
        logger.debug("Computed {} tile predictions".format(len(crop_preds)))
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


def strip_from_baseline(baseline_n2xy: list[tuple[int,int]], height: float, ltrb: tuple[int,int,int,int]=tuple()) -> list[tuple[int,int]]:
    """
    Given a baseline, construct the strip-shaped polygon with given height.

    Args:
        baseline_n2xy (list[tuple[int,int]]): a sequence of (x,y) points.
        height (float): the strip height.
        ltrb (tuple[int,int,int,int]): LTRB constraint of containing region: if not empty (default),
            shift coordinates that would otherwise exceed the region's boundaries.
    Returns:
        list[tuple[int,int]]: a (N,2) clockwise sequence of (x,y) points.
    """
    raw_polygon = strip_from_centerline( np.array( baseline_n2xy )-[0,height/2], height )
    if ltrb:
        return boxed_in( raw_polygon, ltrb ).tolist()
    return raw_polygon.tolist()


def strip_from_centerline(centerline_n2xy: np.ndarray, height: float) -> np.ndarray:
    """
    Given a centerline, construct the strip-shaped polygon with given height.

    Args:
        centerline_n2xy (np.ndarray): a (N,2) sequence of (x,y) points.
        height (float): the strip height.
    Returns:
        np.ndarray: a (N,2) clockwise sequence of (x,y) points.
    """
    left_dummy_pt = np.array( [ 2*centerline_n2xy[0][0]-centerline_n2xy[1][0], 2*centerline_n2xy[0][1]-centerline_n2xy[1][1] ])
    right_dummy_pt = np.array( [ 2*centerline_n2xy[-1][0]-centerline_n2xy[-2][0], 2*centerline_n2xy[-1][1]-centerline_n2xy[-2][1] ])
    centerline_n2xy = np.concatenate( [ [left_dummy_pt], centerline_n2xy, [right_dummy_pt] ], dtype='float')

    vertebras_n2xy = []
    vertebra_north_south_2xy = np.array([[0,-height/2], [0,height/2]])
    for ctr_idx in range(1,len(centerline_n2xy)-1):
        left, mid, right = centerline_n2xy[ctr_idx-1:ctr_idx+2]
        try:
            rotation_matrix = bisection_rotation_matrix( left-mid, right-mid )
            rotated_vertebra_north_south_2xy=np.matmul( rotation_matrix, vertebra_north_south_2xy.T).T
            vertebras_n2xy.append( rotated_vertebra_north_south_2xy + mid ) # shift to actual pos.
        except Exception as e:
            logger.warning(e)
            continue
    vertebras_n2xy = np.stack(vertebras_n2xy)
    contour_pts_n2xy = np.concatenate( [vertebras_n2xy[:,0], vertebras_n2xy[::-1,1], vertebras_n2xy[0:1,0]])
    return contour_pts_n2xy.astype('int32')


def boxed_in( sequence_n2xy: np.ndarray, ltrb: tuple[float,float,float,float] )->np.ndarray:
    """
    Given a sequence of points, shift its elements' coordinates  s.t. they are contained
    within the given box. Can be used for (y,x) points: be sure to pass the box as (t,l,b,r).

    Args: 
        sequence_n2xy (np.ndarray) a (N,2) sequence of (x,y) points.
        ltrb (tuple[float,float,float,float]): the left, top, right, and bottom coordinates.
    Returns:
        polyg_n2xy (np.ndarray): a (N,2) sequence of (x,y) points.
    """
    left, top, right, bottom = ltrb
    shifted_pts = []
    for pt in sequence_n2xy:
        x, y = pt
        if x < left:
            x = left
        elif x > right:
            x = right
        if y < top:
            y = top
        elif y > bottom:
            y = bottom
        shifted_pts.append( [x,y] )
    return np.array( shifted_pts )


def bisection_rotation_matrix(left, right):
    """ Given 2 vectors <left> and <right>, return the matrix that rotates a vertical vector 
    such that it bisects the angle formed by <left> and <right>.

    left (float): a 2D vector/pt.
    right (float): a 2D vector/pt.
    """
    # special case (1): vertical segment
    if np.isclose(left[0], right[0]):
        raise ValueError("Vertical segment: abort.")
    # special case (2): colinear, horizontal vectors
    if np.isclose(left[1], right[1]): 
        return np.identity(2)
    alpha, beta, gamma = 0, 0, 0
    if left[0] == 0 and right[0] != 0: # L vector is horizontal
        beta = np.arctan(right[1]/right[0])
        gamma = beta/2
    elif right[0] == 0 and left[0] != 0: # R vector is horizontal
        alpha = np.arctan(left[1]/left[0]) 
        gamma = -alpha/2
    else:
        alpha = np.arctan(left[1]/left[0]) 
        beta = np.arctan(right[1]/right[0]) 
        gamma = ( alpha + beta ) / 2
    cosg, sing = np.cos( gamma ), np.sin( gamma )
    rotation_matrix = np.array([[cosg, -sing],[sing, cosg]])
    return rotation_matrix


def prune_skeleton( skeleton_hw: np.ndarray, left_to_right=True )->np.ndarray:
    """
    Given a binary skeleton tree, find a longest path, as a sequence of pixels.
    (A crude way to prune a skeleton.)

    Args:
        skeleton_hw (np.ndarray): a binary skeleton.
        left_to_right (np.ndarray): ensure that a longest path can only go right, up, or down.

    Returns:
        Tuple[np.ndarray, np.ndarray]: a pair with 
            - (H,W) pruned skeleton (i.e. no branching)
            - (N,2) list of skeleton coordinates 
    Raises:
        ValueError
    """
    def neighborhood( pixel, left_to_right=left_to_right ):
        max_h, max_w = skeleton_hw.shape
        offsets = np.array([[-1,0],[-1,1],[0,1],[1,0],[1,1]]) if left_to_right else np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]])
        return [ nb for nb in pixel+offsets if (nb[0]>=0 and nb[0]<max_h and nb[1]>=0 and nb[1]<max_w and skeleton_hw[nb[0], nb[1]]) ] 

    # Find left-most pixel with neighborhood 1 (root of the tree)
    min_x, leftmost = 2**10, None
    for px in np.stack( np.where( skeleton_hw == True ), axis=1 ):
        if len(neighborhood( px )) > 1:
            continue
        if px[1] < min_x:
            min_x = px[1]
            leftmost = px

    max_h, max_w = skeleton_hw.shape
    parent_matrix = [ [ None ] * max_w for i in range(max_h) ]
    depth_matrix = np.zeros( skeleton_hw.shape )

    def dfs_iterative( px ):
        """ Iterative DFS, to circumvent recursion limits."""
        visited_matrix = [ [False] * max_w for i in range(max_h) ] 
        current = leftmost
        while(1):
            neighbors = neighborhood( current )
            if not neighbors or all( [ visited_matrix[ nb[0]][nb[1]] for nb in neighbors ] ):
                if current is leftmost:
                    break
                current = parent_matrix[ current[0] ][ current[1] ]
            for nb in neighbors:
                y,x = nb
                if visited_matrix[y][x]:
                    continue
                visited_matrix[y][x]=True
                parent_matrix[y][x]=current
                depth_matrix[y][x]=depth_matrix[ current[0] ][ current[1] ]+1
                current = nb
                break

    dfs_iterative( leftmost )
    deepest_leaf = np.stack( np.unravel_index( np.argmax(depth_matrix), depth_matrix.shape ))

    # New skeleton is a longest path
    longest_path = []
    current = deepest_leaf
    pruned_skeleton, skeleton_coords_n2 = None, None
    while ( not np.array_equal(current, leftmost )):
        longest_path.append( parent_matrix[current[0]][current[1]] )
        current = longest_path[-1]
    #try:
    skeleton_coords_n2 = np.stack([ px for px in longest_path[::-1]], axis=0)
    rr, cc = skeleton_coords_n2.transpose()
    pruned_skeleton = np.zeros( skeleton_hw.shape )
    pruned_skeleton[rr,cc]=1
    #except ValueError e:
    #    loggger.error("Polygonization failed on connected component. Abort.")
        
    return ( pruned_skeleton, skeleton_coords_n2 )


def polygon_to_mask_pil( size: tuple, coords_n2: np.ndarray) -> np.ndarray:
    """
    A small routine to substitute to ski.draw.polygon2mask, whose polygon-filling routine
    is (on some inputs) terribly and misteriously inefficient.

    Args:
        size (tuple[int,int]): image array size.
        coords_n2 (np.ndarray): list of of coordinates (order should match the image size).

    Returns:
        np.ndarray: a (H,W) binary array.
    """
    img = Image.new('1', size=size)
    ImageDraw.Draw( img ).polygon( coords_n2.flatten().tolist(), outline=1,fill=(1,) )
    polyg_hw = np.array( img, dtype='bool').T
    # returning the original object or even a copy of it results in a mysterious
    # segmentation fault when trying to skeletonize
    return polyg_hw + np.zeros( polyg_hw.shape )


def line_count_estimate( img: Union[Image.Image,np.ndarray], sample_width=300, repeat=3) -> int:
    """
    Use FFT to compute a line count estimate for the image.

    Args:
        img (Union[Image.Image,np.ndarray]): an (W,H,C) image or (H,W,C) array.
        sample_width (int): width of the vertical strip whose FG pixel projection should be used
            for FFT.

    Returns:
        int: an estimate of the line count; return -1 if image is too small with respect to the
            sample_width.
    """
    img_hwc = np.array( img ) if isinstance( img, Image.Image) else img
    if img_hwc.shape[1] < sample_width * 1.5:
        return -1
    freq_maxs = []
    img_binary_mask = seglib.get_binary_mask(img_hwc)
    for offset in [ random.randrange( img_hwc.shape[1]-sample_width ) for i in range(repeat) ]:
        vertical_projection = np.sum(np.array( img_binary_mask[10:-10, offset:offset+sample_width]), axis=1)
        vert_fft = np.fft.fft( vertical_projection )
        power_spectrum = np.abs( vert_fft )**2
        freq_ps = list(zip( np.fft.fftfreq( len(vertical_projection) ), power_spectrum ))
        freq_maxs.append( max([ (f,p) for (f,p) in freq_ps if f > 0], key=lambda x: x[1] )[0] )
    return int(len(vertical_projection)*np.mean(freq_maxs))

    
