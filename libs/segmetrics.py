
#stdlib
from pathlib import Path
from typing import Callable, Optional, Union, Mapping, Any
import itertools
import sys
import copy

# 3rd-party
from PIL import Image, ImageDraw
import torch
from torch import Tensor
from torchvision.tv_tensors import Mask
import numpy as np
import numpy.ma as ma


"""Functions for evaluating segmentation outputs.

+ storing polygons on tensors (from dictionaries or pageXML outputs)
+ computing IoU and F1 scores over GT/predicted label maps

A note about types:

+ PageXML or JSON: initial input (typically: from segmentation framework)
+ torch.Tensor: map storage and computations (eg. counting intersections)
+ np.ndarray: metrics and scores; initial mapping of labels: use 32-bit _signed_ integers
  for storing compound (intersecting) labels, to ensure smooth conversion into
  tensors.
"""

def apply_polygon_mask_to_map(label_map: np.ndarray, polygon_mask: np.ndarray, label: int) -> None:
    """In the segmentation map, label pixels matching a given polygon. Up to 4 labels
    can be stored on a single pixel. A label cannot be applied twice to the same
    map.

    Args:
        label_map (np.ndarray): the map that stores the polygons
        polygon_mask (np.ndarray): a binary mask representing the polygon to be labeled.
        label (int): label for this polygon; if the pixel already has a previous label l', the resulting,
            compound value is `(l'<<8)+label`. Eg. 1. Applying label 4 on a pixel that
            already stores label 2 yields `2 << 8 + 4 = 0x204 = 8192`
            Eg. 2. Pixel ``0x10403`` stores labels ``[1, 4, 3]``

    """
    label_limit = 0xff
    max_three_polygon_label = 0xffffff

    # a label may not use more than 1 byte.
    # With large label values, 4-polygon intersections may result in negative compound label values, though,
    # which is not an issue (the map is meant to be stored and used as an cube of unsigned bytes).
    if label > label_limit:
        raise OverflowError('Overflow: label value ({}) exceeds the limit ({}).'.format( label, label_limit ))

    # Handling duplicated labels:
    if array_has_label(label_map, label):
        raise ValueError("The label map already contains a label with value ({})".format(label))

    # for every pixel in intersection...
    intersection_boolean_mask = np.logical_and( label_map, polygon_mask )
    # if intersection does not already contain a 3-polygon pixel
    if np.any( label_map[ intersection_boolean_mask ] > max_three_polygon_label ):
        maxed_out_pixels = np.transpose(((label_map * intersection_boolean_mask) > max_three_polygon_label).nonzero())
        raise ValueError('Cannot store more than 4 polygons on the same pixel! Following positions maxed out: {}{}'.format(
            repr([ (row,col) for (row,col) in maxed_out_pixels ][:5]),
            ' ...' if len(maxed_out_pixels)>5 else ''))
    # ... shift it
    label_map[ intersection_boolean_mask ] <<= 8

    # only then add label to all pixels matching the polygon
    label_map += polygon_mask.astype( label_map.dtype ) * label


def array_to_rgba_uint8( img_hw: np.ndarray ) -> Tensor:
    """Converts a numpy array of 32-bit integers into a 4-channel tensor.

    Args:
        img_hw (np.ndarray): a flat label map of 32-bit integers.

    Returns:
        Tensor: a 4-channel (c,h,w) tensor of unsigned 8-bit integers.
    """
    if len(img_hw.shape) != 2:
        raise TypeError(format("Input map should have shape (W,H) (actual: {}).".format( img_hw.shape )))
    if img_hw.dtype != 'int32': 
        raise TypeError("Label map's dtype should 'int32' (actual: {}".format( img_hw.dtype ))
    img_hw_32b = img_hw.astype('int32')
    img_chw = torch.from_numpy( np.moveaxis( img_hw_32b.view(np.uint8).reshape( (img_hw.shape[0], -1, 4)), 2, 0))
    return img_chw


def polygon_pixel_metrics_from_img_segmentation_dict(img_whc: Image.Image, segmentation_dict_pred: dict, segmentation_dict_gt: dict, binary_mask: Optional[Tensor]=None) -> np.ndarray:
    """Compute a IoU matrix from an image and two dictionaries describing the segmentation's output (line polygons).

    Args:
        img_whc (Image.Image): the input page, needed for the binarization mask.
        segmentation_dict_pred (dict): a dictionary, typically
            constructed from a JSON file.
        segmentation_dict_gt (dict): a dictionary, typically constructed
            from a JSON file.

    Returns:
        np.ndarray: a 2D array, representing IoU values for each possible pair of polygons.
    """
    #polygon_img_gt: Tensor, polygon_img_pred: Tensor, mask: Tensor) -> Tensor:
    polygon_chw_pred, polygon_chw_gt = [ polygon_map_from_segmentation_dict( d ) for d in (segmentation_dict_pred, segmentation_dict_gt) ]

    binary_mask = get_binary_mask( img_whc )

    return polygon_pixel_metrics_from_polygon_maps_and_mask( polygon_chw_pred, polygon_chw_gt, binary_mask )



def polygon_pixel_metrics_from_polygon_maps_and_mask(polygon_chw_pred: Tensor, polygon_chw_gt: Tensor, binary_hw_mask: Optional[Tensor]=None, label_distance=0) -> np.ndarray:
    """Compute pixel-based metrics from two tensors that each encode (potentially overlapping) polygons
    and a FG mask.

    Args:
        polygon_chw_gt (Tensor): a 4-channel image, where each position may store up
            to 3 overlapping labels (one for each channel)
        polygon_chw_pred (Tensor): a 4-channel image, where each position may store up
            to 3 overlapping labels (one for each channel)
        binary_hw_mask (Tensor): a boolean mask that selects the input image's FG pixel

    Returns:
        np.ndarray: metrics (intersection, union, precision, recall, f1) values for each
            possible pair of labels (i,j) with i ∈  map1 and j ∈ map2. Shared pixels in
            each map (i.e. overlapping polygons) have their weight decreased according to the
            number of polygons they vote for.
    """

    if binary_hw_mask is None:
        binary_hw_mask = torch.full( polygon_chw_gt.shape[1:], 1, dtype=torch.bool )
    if binary_hw_mask.shape != polygon_chw_gt.shape[1:]:
        print("mask shape=", binary_hw_mask.shape, "polygon_chw_gt.shape =", polygon_chw_gt.shape)
        raise TypeError("Wrong type: binary mask should have shape {}".format(polygon_chw_gt.shape[1:]))

    if len(polygon_chw_gt.shape) != 3 or polygon_chw_gt.shape[0] != 4 or polygon_chw_gt.dtype is not torch.uint8:
        raise TypeError("Wrong type: polygon GT map should be a 4-channel tensor of unsigned 8-bit integers.")
    if len(polygon_chw_pred.shape) != 3 or polygon_chw_pred.shape[0] != 4 or polygon_chw_pred.dtype is not torch.uint8:
        raise TypeError("Wrong type: polygon predicted map should be a 4-channel tensor of unsigned 8-bit integers.")
    if polygon_chw_gt.shape != polygon_chw_pred.shape:
        raise TypeError("Wrong type: both maps should have the same shape (instead: {} and {}).".format( polygon_chw_gt.shape, polygon_chw_pred.shape ))

    polygon_chw_fg_pred, polygon_chw_fg_gt = [ polygon_img * binary_hw_mask for polygon_img in (polygon_chw_pred, polygon_chw_gt) ]
    
    metrics = polygon_pixel_metrics_two_deep_maps( polygon_chw_fg_pred, polygon_chw_fg_gt, label_distance )

    return metrics

def polygon_pixel_metrics_two_flat_maps_and_mask(map_hw_1: Tensor, map_hw_2: Tensor, binary_hw_mask: Optional[Tensor]=None, label_distance=0) -> np.ndarray:

    """Compute pixel-based metrics from two tensors that each encode non-overlapping polygons
    and a FG mask.

    Args:
        map_hw_1 (np.ndarray): the predicted map, i.e. a flat map of labeled polygons.
        map_hw_2 (np.ndarray): the GT map, i.e. a flat map of labeled polygons.
        binary_hw_mask (Tensor): a boolean mask that selects the input image's FG pixel

    Returns:
        np.ndarray: metrics (intersection, union, precision, recall, f1) values for each
            possible pair of labels (i,j) with i ∈  map1 and j ∈ map2. Shared pixels in
            each map (i.e. overlapping polygons) have their weight decreased according to the
            number of polygons they vote for.
    """

    if binary_hw_mask is None:
        binary_hw_mask = torch.full( map_hw_1.shape, 1, dtype=torch.bool )
    #print("map_hw_1.shape:", map_hw_1.shape, "map_hw_2.shape:", map_hw_2.shape)
    if binary_hw_mask.shape != map_hw_2.shape:
        print("mask shape=", binary_hw_mask.shape, "map_hw_2.shape =", map_hw_2.shape)
        raise TypeError("Wrong type: binary mask should have shape {}".format(map_hw_2.shape))

    map_hw_fg_1, map_hw_fg_2 = [ mp * binary_hw_mask for mp in (map_hw_1, map_hw_2) ]
    metrics = polygon_pixel_metrics_two_flat_maps( map_hw_fg_1, map_hw_fg_2, label_distance )

    return metrics

def retrieve_polygon_mask_from_map( label_map_chw: Tensor, label: int) -> Tensor:
    """From a label map (that may have compound pixels representing polygon intersections),
    compute a binary mask that covers _all_ pixels for the label, whether they belong to an
    intersection or not.

    Args:
        label_map_chw (Tensor): a 4-channel tensor, where each pixel can store up to 4 labels.
        label (int): the label to be selected.

    Returns:
        Tensor: a flat, boolean mask for the polygon of choice.
    """
    if len(label_map_chw.shape) != 3 and label_map_chw.shape[0] != 4:
        raise TypeError("Wrong type: label map should be a 4-channel tensor (shape={} instead).".format( label_map_chw.shape ))
    polygon_mask_hw = torch.sum( label_map_chw==label, dim=0).type(torch.bool)

    return polygon_mask_hw


def array_has_label( label_map_hw: np.ndarray, label: int ) -> bool:
    """From a flat label map (as generated from a segmentation dictionary) where each pixel can store up to 3 values,
    test whether a given polygon has been stored already.

    Args:
        label_map_hw (np.ndarray): a 2D map, where each 32-bit integer store up to 4 labels.
        label (int): the label to be checked for.

    Returns:
        bool: True if map already stores the given label; False otherwise.
    """
    if len(label_map_hw.shape) > 2:
        raise TypeError("Map should be a flat map of integers.")

    label_cube_chw = np.moveaxis(label_map_hw.view('uint8').reshape(label_map_hw.shape+(-1,)), 2, 0)
    return bool(np.any( label_cube_chw == label ))

def polygon_pixel_metrics_two_flat_maps( map_hw_1: np.ndarray, map_hw_2: np.ndarray, label_distance=5) -> np.ndarray:
    """Provided two label maps that each encode _non-overlapping_ polygons, compute
    for each possible pair of labels (i_pred, j_gt) with i ∈  map1 and j ∈  map2.
    + intersection and union counts
    + precision and recall

    Args:
        map_hw_1 (np.ndarray): the predicted map, i.e. a flat map of labeled polygons.
        map_hw_2 (np.ndarray): the GT map, i.e. a flat map of labeled polygons.

    Returns:
        np.ndarray: a 4 channel array, where each cell [i,j] stores respectively intersection and union
            counts, as well as precision and recall for a pair of labels [i,j].
    """
    min_label_1, max_label_1 = int(np.min( map_hw_1[ map_hw_1 > 0 ] ).item()), int(np.max( map_hw_1 ).item())
    min_label_2, max_label_2 = int(np.min( map_hw_2[ map_hw_2 > 0 ] ).item()), int(np.max( map_hw_2 ).item())
    label2index_1 = { l:i for i,l in enumerate( range(min_label_1, max_label_1+1)) }
    label2index_2 = { l:i for i,l in enumerate( range(min_label_2, max_label_2+1)) }
    metrics_hwc = np.zeros(( max_label_1-min_label_1+1, max_label_2-min_label_2+1, 4), dtype='float32')
    label_range_1, label_range_2 = range(min_label_1, max_label_1+1), range(min_label_2, max_label_2+1)
    #print("Ranges: ({}->{}), ({}->{})".format(min_label_1, max_label_1, min_label_2, max_label_2))

    # retrieve individual masks for each label and stack them up
    label_matrices_1={ l:(map_hw_1 == l) for l in label_range_1 }
    label_matrices_2={ l:(map_hw_2 == l) for l in label_range_2 }

    #print(label_range_1, label_range_2)
    label_counts_1 = { l:np.sum(label_matrices_1[l]).item() for l in label_range_1 }
    label_counts_2 = { l:np.sum(label_matrices_2[l]).item() for l in label_range_2 }

    for lbl1, lbl2 in itertools.product(label_range_1, label_range_2):
        if label_distance > 0 and abs(lbl1-lbl2) > label_distance:
            metrics_hwc[label2index_1[lbl1], label2index_2[lbl2]]=[ 0, label_counts_1[lbl1] + label_counts_2[lbl2], 0, 0 ]
            continue
        # intersection
        intersection_count = np.sum(label_matrices_1[ lbl1 ] * label_matrices_2[ lbl2 ]).item()
        label_1_count, label_2_count = np.sum(label_matrices_1[lbl1]), np.sum(label_matrices_2[lbl2])  
        union_count = label_1_count + label_2_count - intersection_count
        # precision: true pred / all pred
        if label_1_count != 0:
            precision = intersection_count / label_1_count
        # recall: true pred / all gt
        if label_2_count !=0:
            recall = intersection_count / label_2_count
        metrics_hwc[label2index_1[lbl1], label2index_2[lbl2]] = [intersection_count, union_count, precision, recall ]
    return metrics_hwc


@torch.no_grad()
def polygon_pixel_metrics_two_flat_maps_torch( map_hw_1: np.ndarray, map_hw_2: np.ndarray, label_distance=5, device='cpu') -> np.ndarray:
    """Provided two label maps that each encode _non-overlapping_ polygons, compute
    for each possible pair of labels (i_pred, j_gt) with i ∈  map1 and j ∈  map2.
    + intersection and union counts
    + precision and recall
    This is the torch version.

    Args:
        map_hw_1 (np.ndarray): the predicted map, i.e. a flat map of labeled polygons.
        map_hw_2 (np.ndarray): the GT map, i.e. a flat map of labeled polygons.

    Returns:
        np.ndarray: a 4 channel array, where each cell [i,j] stores respectively intersection and union
            counts, as well as precision and recall for a pair of labels [i,j].
    """
    map_hw_1 = torch.tensor( map_hw_1 ).to(device)
    map_hw_2 = torch.tensor( map_hw_2 ).to(device)

    min_label_1, max_label_1 = int(torch.min( map_hw_1[ map_hw_1 > 0 ] ).item()), int(torch.max( map_hw_1 ).item())
    min_label_2, max_label_2 = int(torch.min( map_hw_2[ map_hw_2 > 0 ] ).item()), int(torch.max( map_hw_2 ).item())
    label2index_1 = { l:i for i,l in enumerate( range(min_label_1, max_label_1+1)) }
    label2index_2 = { l:i for i,l in enumerate( range(min_label_2, max_label_2+1)) }
    metrics_hwc = torch.zeros(( max_label_1-min_label_1+1, max_label_2-min_label_2+1, 4), dtype=torch.float32)
    label_range_1, label_range_2 = range(min_label_1, max_label_1+1), range(min_label_2, max_label_2+1)
    #print("Ranges: ({}->{}), ({}->{})".format(min_label_1, max_label_1, min_label_2, max_label_2))

    # retrieve individual masks for each label and stack them up
    label_matrices_1={ l:(map_hw_1 == l) for l in label_range_1 }
    label_matrices_2={ l:(map_hw_2 == l) for l in label_range_2 }

    #print(label_range_1, label_range_2)
    label_counts_1 = { l:torch.sum(label_matrices_1[l]).item() for l in label_range_1 }
    label_counts_2 = { l:torch.sum(label_matrices_2[l]).item() for l in label_range_2 }

    for lbl1, lbl2 in itertools.product(label_range_1, label_range_2):
        if label_distance > 0 and abs(lbl1-lbl2) > label_distance:
            metrics_hwc[label2index_1[lbl1], label2index_2[lbl2]]=torch.tensor([ 0, label_counts_1[lbl1] + label_counts_2[lbl2], 0, 0 ])
            continue
        # intersection
        intersection_count = torch.sum(label_matrices_1[ lbl1 ] * label_matrices_2[ lbl2 ]).item()
        label_1_count, label_2_count = torch.sum(label_matrices_1[lbl1]), torch.sum(label_matrices_2[lbl2])  
        union_count = label_1_count + label_2_count - intersection_count
        # precision: true pred / all pred
        if label_1_count != 0:
            precision = intersection_count / label_1_count
        # recall: true pred / all gt
        if label_2_count !=0:
            recall = intersection_count / label_2_count
        metrics_hwc[label2index_1[lbl1], label2index_2[lbl2]] = torch.tensor([intersection_count, union_count, precision, recall ])
    return np.asarray( metrics_hwc )


def polygon_pixel_metrics_two_deep_maps( map_chw_1: Tensor, map_chw_2: Tensor, label_distance=5) -> np.ndarray:
    """Provided two label maps that each encode (potentially overlapping) polygons, compute
    for each possible pair of labels (i_pred, j_gt) with i ∈  map1 and j ∈  map2.
    + intersection and union counts
    + precision and recall

    Shared pixels in each map (i.e. overlapping polygons) have their weight decreased according
    to the number of polygons they vote for.

    Args:
        map_chw_1 (Tensor): the predicted map, i.e. a 4-channel map with labeled polygons, with potential overlaps.
        map_chw_2 (Tensor): the GT map, i.e. a 4-channel map with labeled polygons, with potential overlaps.

    Returns:
        np.ndarray: a 4 channel array, where each cell [i,j] stores respectively intersection and union
            counts, as well as precision and recall for a pair of labels [i,j].
    """
    min_label_1, max_label_1 = int(torch.min( map_chw_1[ map_chw_1 > 0 ] ).item()), int(torch.max( map_chw_1 ).item())
    min_label_2, max_label_2 = int(torch.min( map_chw_2[ map_chw_2 > 0 ] ).item()), int(torch.max( map_chw_2 ).item())
    label2index_1 = { l:i for i,l in enumerate( range(min_label_1, max_label_1+1)) }
    label2index_2 = { l:i for i,l in enumerate( range(min_label_2, max_label_2+1)) }

    # 4 channels for the intersection and union counts, and the precision and recall scores, respectively
    metrics_hwc = np.zeros(( max_label_1-min_label_1+1, max_label_2-min_label_2+1, 4), dtype='float32')

    label_range_1, label_range_2 = range(min_label_1, max_label_1+1), range(min_label_2, max_label_2+1)
    
    # Factor out some computations
    label_matrices_1={ l:retrieve_polygon_mask_from_map( map_chw_1, l) for l in label_range_1 }
    label_matrices_2={ l:retrieve_polygon_mask_from_map( map_chw_2, l) for l in label_range_2 }
    depth_1, depth_2 = map_to_depth( map_chw_1 ), map_to_depth( map_chw_2 )
    max_depth = torch.max( depth_1, depth_2 ) # for each pixel, keep the largest depth value of the two maps
    label_counts_1={ l:torch.sum(label_matrices_1[l]/depth_1).item() for l in label_range_1 }
    label_counts_2={ l:torch.sum(label_matrices_2[l]/depth_2).item() for l in label_range_2 }

    for lbl1, lbl2 in itertools.product(label_range_1, label_range_2):
        # assume that labels beyond a given distance do not intersect
        if label_distance > 0 and abs(lbl1-lbl2) > label_distance:
            metrics_hwc[label2index_1[lbl1], label2index_2[lbl2]]=[ 0, label_counts_1[lbl1] + label_counts_2[lbl2], 0, 0 ]
            continue
        # Idea:
        # + the intersection of a label 1 of depth m (where m = # of polygons that intersect on the pixel) with label 2
        # of depth n has weight 1/max(m, n)
        # + the union of a label-1 pixel of depth m with label-2 pixel of depth n has weight (1/m + 1/n)

        # 1. Compute intersection boolean matrix
        #print("1. Compute intersection boolean matrix, depth1 and depth2")
        label_1_matrix, label_2_matrix = label_matrices_1[ lbl1 ], label_matrices_2[ lbl2 ]
        intersection_mask = label_1_matrix * label_2_matrix

        # 2. Compute the weighted intersection count of the two maps
        #print("3. For each pixel, keep the largest depth value of the two maps")
        intersection_count = torch.sum( intersection_mask / max_depth ).item()

        # 3. Compute cardinalities |label 1| and |label 2| in map 1 and map 2, respectively
        #print('4. Compute cardinalities |label 1| and |label 2| in map 1 and map 2, respectively')
        label_1_count, label_2_count = label_counts_1[lbl1], label_counts_2[lbl2]

        # 4. union = |label 1| + |label 2| - |label 1 ∩ label 2|
        #print('5. union = |label 1| + |label 2| - |label 1 ∩ label 2|')
        union_count = label_1_count + label_2_count - intersection_count

        # 5. P = |label 1 ∩ label 2| / | label 2 |; R = |label 1 ∩ label 2| / | label 1 |
        #print('6. P = |label 1 ∩ label 2| / | label 2 |; R = |label 1 ∩ label 2| / | label 1 |')
        if label_1_count != 0:
            # rows (label_1) assumed to be predictions
            precision = intersection_count / label_1_count
        if label_2_count != 0:
            # cols (label_2) assumed to be GT
            recall = intersection_count / label_2_count

        metrics_hwc[label2index_1[lbl1], label2index_2[lbl2]]=[ intersection_count, union_count, precision, recall ]

    return metrics_hwc


def map_to_depth(map_chw: Tensor) -> Tensor:
    """Compute depth of each pixel in the input map, i.e. how many polygons intersect on this pixel.
    Note: 0-valued pixels have depth 1.

    Args:
        map_chw (Tensor): the input tensor (4 channels)

    Returns:
        Tensor: a tensor of integers, where each value represents the
        number of intersecting polygons for the same position in the
        input map.
    """
    depth_map = torch.sum( map_chw != 0, dim=0)
    depth_map[ depth_map == 0 ]=1

    return depth_map


def polygon_pixel_metrics_to_line_based_scores_icdar_2017( metrics: np.ndarray, threshold: float=.75 ) -> np.ndarray:
    """Implement ICDAR 2017 evaluation metrics, as described in
    https://github.com/DIVA-DIA/DIVA_Line_Segmentation_Evaluator/releases/tag/v1.0.0
    (a Java implementation)

    IoU = TP / (TP+FP+FN)
    F1 = (2*TP) / (2*TP+FP+FN)
 
    + find all polygon pairs that have a non-empty intersection
    + in Pred: labels that are _not_ in a pair above are FP
      in GT: labels that are _not_ in a pair above are FN
    + sort the pairs by IoU
    + traverse the IoU-descending sorted list and select the first available match
      for each polygon belonging to the prediction set (thus ensuring that no
      polygon can be matched twice)
    + a pred/GT match is TP if both Recall and Precision > .75.
    + a pred/GT match is FP if P < .75; FN if R < .75 (it can be both!)
    + Line-based, per-page IoU [or Jaccard index]= TP/(TP+FP+FN)
    + Document-wide, pixel-based IoU_px  = TP_px/TP_px + FP_px + FN_px}.

    Args:
        metrics (np.ndarray): metrics matrix, with indices [0..m-1, 0..n-1] for labels 1..m, 
            where m and n are the max. labels of of the predicted and GT maps respectively.
            In the channels: intersection count, union count, precision, recall.
        threshold (float): IoU threshold for TP

    Returns:
        np.ndarray: a 5-elt array with the TP-, FP-, and FN-counts, as well as the Jaccard (aka. IoU)
            and F1 score at the line level.
    """
    label_count_pred, label_count_gt = metrics.shape[:2]
    #print(label_count_pred, label_count_gt)

    # find all rows with non-empty intersection (excluding background)
    possible_match_indices = metrics[:,:,0].nonzero()
    
    TP = 0.0
    FP = len([ l for l in range(label_count_pred) if l not in possible_match_indices[0]])
    FN = len([ l for l in range(label_count_gt) if l not in possible_match_indices[1]])

    match_rows_cols, possible_matches = np.transpose(possible_match_indices), metrics[ possible_match_indices ]
    ious = possible_matches[:,0]/possible_matches[:,1]
    structured_row_col_match_iou = np.array([
        (match_rows_cols[i,0],
         match_rows_cols[i,1],
         possible_matches[i,0],
         possible_matches[i,1],
         possible_matches[i,2],
         possible_matches[i,3],
         ious[i]) for i in range(len(possible_matches))],
         dtype=[('pred_polygon', 'int32'),
                ('gt_polygon', 'int32'),
                ('intersection', 'float32'),
                ('union', 'float32'),
                ('precision', 'float32'),
                ('recall', 'float32'),
                ('iou', 'float32')])

    #print(structured_row_col_match_iou)
    
    # sort candidate matches by ascending label order and descending IoU order
    pred_label_iou_copy = structured_row_col_match_iou[['pred_polygon', 'iou']].copy()
    pred_label_iou_copy['iou'] *= -1
    I = np.argsort( pred_label_iou_copy, order=['pred_polygon', 'iou'])

    # select one-to-one matches
    pred2match = { i:False for i in possible_match_indices[0] }
    for possible_match in structured_row_col_match_iou[I]:
        # ensure that each predicted label is matched to at most one GT label
        # (first hit is the one with highest IoU)
        if not pred2match[possible_match['pred_polygon']]:
            pred2match[possible_match['pred_polygon']]=True
            precision, recall = possible_match[['precision', 'recall']]
            #print("precision=", precision, "recall=", recall)
            TP += (precision >= threshold and recall >= threshold )
            # a FP is a non-zero (Pred, GT) pair whose P < .75 or: the system detects
            # a polygon that partially capture the GT, but too much of the rest also
            FP += precision < threshold
            # a FN is a non-zero (Pred, GT) pair whose R < .75 or: the system detects
            # a polygon that matches the GT, but not enough of it
            FN += recall < threshold
            #print(TP, FP, FN)

    if TP+FP and TP+FN:
        Precision = TP / (TP+FP) 
        Recall = TP / (TP+FN) 
        Jaccard = TP / (TP+FP+FN)
        F1 = 2*TP / (2*TP+FP+FN)
        return np.array([threshold, TP, FP, FN, Precision, Recall, Jaccard, F1])
    return np.array([ threshold, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan ] )

def polygon_pixel_metrics_to_line_based_scores( metrics_hwc: np.ndarray, threshold: float=.75 ) -> np.ndarray:
    """Classic evalution metrics, where mask are matched based on the best IoU.

    IoU = TP / (TP+FP+FN)
    F1 = (2*TP) / (2*TP+FP+FN)
 
    + find all polygon pairs that have a non-empty intersection
    + sort the pairs by IoU
    + traverse the IoU-descending sorted list and select the first available match
      for each polygon belonging to the prediction set (thus ensuring that no
      polygon can be matched twice)
    + a pred/GT match is TP if IoU > threshold; any unmatched predicted label is FP
    + any unmatched GT label is a FN

    Args:
        metrics (np.ndarray): metrics matrix, with indices [0..m-1, 0..n-1] for labels 1..m, 
            where m and n are the max. labels of of the predicted and GT maps respectively.
            In the channels: intersection count, union count, precision, recall.
        threshold (float): IoU threshold for TP

    Returns:
        np.ndarray: a 3-elt array with the TP-, FP-, and FN-counts.
    """
    label_count_pred, label_count_gt = metrics_hwc.shape[:2]

    # keep only IoU, with preds in rows, gt in cols
    pred_gt_ious = metrics_hwc[:,:,0]/metrics_hwc[:,:,1]
    #print(pred_gt_ious)

    pred_to_gt = {}
    best_match_iou = {}
    FP = 0

    # match each predicted with its unique gt line, based on best IoU
    # TODO
    for lpred in range(label_count_pred):
        for lgt in range(label_count_gt):
            iou = pred_gt_ious[lpred,lgt]
            if iou > threshold:
                pred_to_gt[ lpred ] = lgt
    #print(pred_to_gt)
    # false positives: all those predictions that do not have a match with GT
    FP = len(set(range(label_count_pred)) - set(pred_to_gt.keys()))
    FN = len(set(range(label_count_gt)) - set(pred_to_gt.values()))
    TP = len(pred_to_gt.items())

    if TP+FP and TP+FN:
        Precision = TP / (TP+FP) 
        Recall = TP / (TP+FN) 
        Jaccard = TP / (TP+FP+FN)
        F1 = 2*TP / (2*TP+FP+FN)
        return np.array([threshold, TP, FP, FN, Precision, Recall, Jaccard, F1])

    return np.array([ threshold, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan ] )



def mAP( pixel_metrics_list: list[np.ndarray] ):
    """
    mAP = (AP_.5 + AP.55 + ... + AP.95) / 10

    where
        AP_<thr> is obtained as follow:
        + sort all predicted lines by confidence score (or IoU)
        + for each entry: compute accumulated TP_<thr>, FP_<thr>, Prec, Rec
        + (optional) plot PR curve

    Args:
        pixels_metrics_list (list[np.ndarray]): a list of page-wide, 2-mask pixel metrics, as computed by 
            `polygon_pixel_metrics_two_flat_maps`.
    Return:
        tuple[float,list[tuple[float,float]]]: a tuple with
            - mAP across all IoU thresholds
            - sorted sequence of (precision,recall) values
    """
    av_metrics = np.zeros((len(pixel_metrics_list),10,3)); # dims: samples,thresholds, metrics
    for i, pm in enumerate(pixel_metrics_list):
        for t,thrld in enumerate(np.linspace(.5,.95,10)): 
            av_metrics[i,t] = polygon_pixel_metrics_to_line_based_scores( pm, threshold=thrld )
#            if t == 3 or t==8:
#                print("Map {}, threshold {}: {} -".format(i, thrld, av_metrics[i,t]))
#    print("Metrics shape:", av_metrics.shape)
#    # sum over all samples
    tp_fp_fn = np.sum( av_metrics, axis=0 ) 
    print('TP|FP|FN')
    print(tp_fp_fn)
    # TP/(TP+FP), TP/(TP+FN) for all thresholds 
    recall = tp_fp_fn[:,0] / (tp_fp_fn[:,0]+tp_fp_fn[:,2]) 
    precision = tp_fp_fn[:,0] / (tp_fp_fn[:,0]+tp_fp_fn[:,1]) 
    print("R:", recall)
    print("P:", precision)


    return {
            'mAP50': precision[0], 
            'mAP75': precision[5], 
            'mAP50_95': np.sum(precision)/10, 
            'recall/precision': list(zip( recall.tolist(), precision.tolist()))}
    


def polygon_pixel_metrics_to_pixel_based_scores( metrics: np.ndarray) -> tuple[float, float]:
    """Implement ICDAR 2017 pixel-based evaluation metrics, as described in
    Simistira et al., ICDAR2017, "Competition on Layout Analysis for Challenging Medieval
    Manuscripts", 2017.

    Two versions of the pixel-based IoU metric:
    + Pixel IU takes all pixels of all intersecting pairs into account
    + Matched Pixel IU only takes into account the pixels from the matched lines

    TODO: verify that threshold value is not relevant for this metric.

    Args:
        metrics (np.ndarray): metrics matrix, with indices [0..m-1, 0..m-1] for labels 1..m,
            where m is the maximum label in either GT or predicted maps. In channels: intersection
            count, union count, precision, recall.

    Returns:
        tuple: a pair (Pixel IU, Matched Pixel IU)
    """
    label_count_pred, label_count_gt = metrics.shape[:2]

    # find all rows with non-empty intersection
    possible_match_indices = metrics[:,:,0].nonzero()
    match_rows_cols, possible_matches = np.transpose(possible_match_indices), metrics[ possible_match_indices ]
    ious = possible_matches[:,0]/possible_matches[:,1]
    structured_row_col_match_iou = np.array([
        (match_rows_cols[i,0],
         match_rows_cols[i,1],
         possible_matches[i,0],
         possible_matches[i,1],
         possible_matches[i,2],
         possible_matches[i,3],
         ious[i]) for i in range(len(possible_matches))],
         dtype=[('pred_polygon', 'int32'),
                ('gt_polygon', 'int32'),
                ('intersection', 'float32'),
                ('union', 'float32'),
                ('precision', 'float32'),
                ('recall', 'float32'),
                ('iou', 'float32')])

    # pixel-based, page-wide IoU (over all non-empty intersections)
    intersection_count, union_count = [ np.sum(structured_row_col_match_iou[:][field]) for field in ('intersection', 'union') ]
    pixel_iou = intersection_count / union_count

    # pixel-based, page-wide IoU (over all matched pairs)
    matched_intersection_count, matched_union_count = 0, 0

    # sort candidate matches by ascending label order and descending IoU order
    pred_label_iou_copy = structured_row_col_match_iou[['pred_polygon', 'iou']].copy()
    pred_label_iou_copy['iou'] *= -1
    I = np.argsort( pred_label_iou_copy, order=['pred_polygon', 'iou'])

    pred2match = { i:False for i in possible_match_indices[0] }
    for possible_match in structured_row_col_match_iou[I]:
        # ensure that each predicted label is matched to at most one GT label
        if not pred2match[possible_match['pred_polygon']]:
            pred2match[possible_match['pred_polygon']]=True
            matched_intersection_count += possible_match['intersection']
            matched_union_count += possible_match['union']
    matched_pixel_iou = matched_intersection_count / matched_union_count

    return (pixel_iou, matched_pixel_iou)


#def metrics_to_precision_recall_curve( metrics: np.ndarray, threshold_range=np.linspace(0, 1, num=21)) -> np.ndarray:
#    """
#    Compute precision and recalls over a range of IoU thresholds, for plotting purpose.
#
#    Args:
#        metrics (np.ndarray): a 4-channel table with GT labels in rows and predicted labels in columns, where
#                              each entry is a [intersection_count, union_count, precision, recall] sequence.
#        threshold_range: a series of threshold values, between 0 and 1 (default: [0, 0.05, 0.1, ..., 0.95, 1])
#
#    :returns:
#        np.ndarray: a 2D array, with precisions in row 0 and recalls in row 1.
#
#    """
#    precisions_recalls = np.zeros((len(threshold_range), 2))
#    for (i,t) in enumerate(threshold_range):
#        precisions_recalls[i] = metrics_to_aggregate_scores(metrics, iou_threshold=t)[:2]
#        #print(precisions_recalls[:,i])
#    return np.moveaxis( precisions_recalls, 1, 0)


def recover_labels_from_map_value( px: int) -> list:
    """Retrieves intersecting polygon labels from a single map pixel value (for
    diagnosis purpose).

    Args:
        vl (int): a map pixel, whose value is a 32-bit signed integer.

    Returns:
        list: a list of labels
    """
    return [ b for b in np.array( [px], dtype='int32').view('uint8')[::-1] if b ]


def mask_from_polygon_map_functional( polygon_map: Tensor, test: Callable) -> Tensor:
    """Given a 3D map of polygons (where each pixel contains at most 4 labels,
    select labels based on a boolean function.
    Eg. ``mask_from_functional( polygon_map, lambda m: m % 2 )`` covers all odd-labeled
        polygons.

    Args:
        polygon_map (Tensor): polygon set, encoded as a 4-channel, 8-bit tensor.
        test (Callable): a boolean function, to be applied to the map; a partial function 
            may be passed, if added parameters are needed.

    Returns:
        Tensor: a boolean, flat mask.
    """
    if polygon_map.dtype != torch.uint8:
        raise TypeError("First parameter should be a Tensor of uint8.")
    if len(polygon_map.shape) != 3 or polygon_map.shape[0]!=4:
        raise TypeError("Polygon map should have shape (4, m, n)")

    return torch.sum( test( polygon_map ), dim=0).type(torch.bool)



def gt_masks_to_labeled_map( masks: Mask ) -> np.ndarray:
    """
    Combine stacks of GT line masks (as in data annotations) into a single, labeled page-wide map.
    """
    return np.sum( np.stack([ m * lbl for (lbl,m) in enumerate(masks, start=1)]), axis=0)


