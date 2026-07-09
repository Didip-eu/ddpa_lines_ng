
#stdlib
from pathlib import Path
import json
from typing import Callable, Optional, Union, Mapping, Any
import itertools
import re
import sys
import math
import copy
from datetime import datetime

# 3rd-party
from PIL import Image, ImageDraw
import skimage as ski
import torch
from torch import Tensor
import numpy as np
from segtformats import segtformats as sgf

# local
from . import polygon_utils

"""
Any routine that involves joint manipulation of images and segmentation metadata.

+ for internal manipulation and conversion of various formats (XML, JSON, Alto) → segformats.py
+ for segmentation evaluation routines → segmetrics.py
"""


def polygon_map_from_json_file(  segmentation_json: str) -> Tensor:
    """Read line polygons from a JSON file and store them into a tensor, as pixel maps.
    Channels allow for easy storage of overlapping polygons.

    Args:
        segmentation_json (str): path of a JSON file

    Returns:
        Tensor: the polygons rendered as a 4-channel image (a tensor).
    """
    with open( segmentation_json, 'r' ) as json_file:
        return polygon_map_from_segmentation_dict( json.load( json_file ))


def polygon_map_from_page_xml_file( page_xml: str ) -> Tensor:
    """Read line polygons from a PageXML file and store them into a tensor, as pixel maps.
    Channels allow for easy storage of overlapping polygons.

    Args:
        page_xml (str): path of a PageXML file.

    Returns:
        Tensor: the polygons rendered as a 4-channel image (a tensor).
    """

    segmentation_dict = sgf.segmentation_dict_from_page_xml( page_xml )
    return polygon_map_from_segmentation_dict( segmentation_dict)

def polygon_map_from_segmentation_dict( segmentation_dict: dict ) -> Tensor:
    """Store line polygons into a tensor, as pixel maps.

    Args:
        segmentation_dict (dict): kraken's segmentation output, i.e. a dictionary of the form::

                 {
                 'image_width': w, 
                 'image_height': h,
                 'text_direction': '$dir',
                 'type': 'baseline',
                 'lines': [
                   {'baseline': [[x0, y0], [x1, y1], ...], 'coords': [[x0, y0], [x1, y1], ... [x_m, y_m]]},
                   ...
                 ]
                 'regions': [ ... ] }

    Returns:
        Tensor: the polygons rendered as a 4-channel image.
    """
    #polygon_boundaries = [ line[polygon_key] for line in segmentation_dict['lines'] ]
    polygon_boundaries = line_polygons_from_segmentation_dict( segmentation_dict )

    # create 2D matrix of 32-bit integers
    # (fillPoly() only accepts signed integers - risk of overflow is non-existent)
    mask_size = segmentation_dict['image_height'], segmentation_dict['image_width']

    label_map = np.zeros( mask_size, dtype='int32' )

    # rendering polygons
    for lbl, polyg in enumerate( polygon_boundaries ):
        points = np.array(polyg)[:,::-1] # x <-> y
        polyg_mask = ski.draw.polygon2mask( mask_size, points )
        apply_polygon_mask_to_map( label_map, polyg_mask, lbl+1 )

    #Image.fromarray( label_map ).show()

    # 8-bit/pixel, 4 channels (note: order is little-endian)
    polygon_img = array_to_rgba_uint8( label_map )
    #ski.io.imshow( polygon_img.permute(1,2,0).numpy() )

    return polygon_img


def line_binary_mask_from_json_file( segmentation_json: str, channels=1 ) -> Tensor:
    """From a JSON segmentation file, return a boolean mask where any pixel belonging
    to a polygon is 1 and the other pixels 0.

    Args:
        segmentation_json (str): a JSON file describing the lines.
        channels (int): number of channels.

    Returns:
        Tensor: a flat boolean tensor with size (H,W)
    """
    with open( segmentation_json, 'r' ) as json_file:
        return line_binary_mask_from_segmentation_dict( json.load( json_file ), channels=channels)


def line_binary_mask_from_page_xml_file( page_xml: str, channels=1 ) -> Tensor:
    """From a PageXML file describing polygons, return a boolean mask where any pixel belonging
    to a polygon is 1 and the other pixels 0.

    Args:
        page_xml (str): a Page XML file describing the lines.
        channels (int): number of channels.

    Returns:
        Tensor: a flat boolean tensor with size (H,W)
    """
    segmentation_dict = sgf.segmentation_dict_from_page_xml( page_xml )
    return line_binary_mask_from_segmentation_dict( segmentation_dict, channels=channels )


def line_binary_mask_from_segmentation_dict( segmentation_dict: dict, channels=1 ) -> Tensor:
    """From a segmentation dictionary describing polygons, return a boolean mask where any pixel belonging
    to a polygon is 1 and the other pixels 0.

    Args:
        segmentation_dict (dict): a dictionary, typically constructed from a JSON file.
        channels (int): number of channels (default is 1).

    Returns:
        Tensor: a flat boolean tensor with size (H,W)
    """
    polygon_boundaries = line_polygons_from_segmentation_dict( segmentation_dict )
    # create 2D boolean matrix
    mask_size = (segmentation_dict['image_width'], segmentation_dict['image_height'])
    one_channel_mask = np.sum( [ ski.draw.polygon2mask( mask_size, polyg ).transpose(1,0) for polyg in polygon_boundaries ], axis=0)
    if channels > 1:
        return torch.tensor( np.tile( one_channel_mask, channels ).reshape( one_channel_mask.shape + (channels,)))
    return torch.tensor( one_channel_mask )


def didip_json_to_label_mask( segmentation_json: str, channels=3, largest_dimension=1248, output_file_path='', overwrite_existing=False ):
    """Convert a DiDip JSON segmentation file into a Doc-UFCN label mask, which 
    is a line mask meeting a size constraint.

    Args:
        segmentation_json (str): path to a segmentation file, DiDip-style
        channels (int): number of channels (default: 3)
        largest_dimension (int): the resulting mask's largest dimension.
        output_file_path (str): output file; if empty, use the standard output.
        overwrite_existing (bool): if False, do not write over older masks.
    """
    with open( segmentation_json, 'r' ) as json_file:
        segmentation_dict = json.load( json_file )
        polygon_boundaries = line_polygons_from_segmentation_dict( segmentation_dict)
        mask_size = (segmentation_dict['image_width'], segmentation_dict['image_height'])
        one_channel_img = Image.fromarray( np.uint8( np.sum( [ ski.draw.polygon2mask( mask_size, polyg ).transpose(1,0) for polyg in polygon_boundaries ], axis=0)), mode="L")
        new_width, new_height = (largest_dimension/mask_size[1] * mask_size[0], largest_dimension)
        if new_width > largest_dimension:
            new_width, new_height = (largest_dimension, largest_dimension/mask_size[0] * mask_size[1])

        img_array = np.array( one_channel_img.resize( (int(new_width), int(new_height)) ))
        if channels>1:
            img_array = np.repeat( img_array, channels ).reshape( img_array.shape + (channels,))
        if output_file_path and overwrite_existing:
            pil_img = Image.fromarray( img_array*255, mode='RGB' if channels==3 else "L" )
            pil_img.save( output_file_path )
        else:
            return np.array( img_array )

def didip_json_to_docufcn_label_json( segmentation_json: Union[Path,dict], output_file_path='', overwrite_existing=False)->dict:
    """ Convert a DiDip JSON segmentation file into a Doc-UFCN label file.
    Note: assumes a single-region file; only for evaluation use in segmentation pipeline.

    Args:
        segmentation_json (Union[Path,dict]): path to a segmentation file, DiDip-style, or the dictionary itself
        output_file_path (str): output file; if empty, use the standard output.
        overwrite_existing (bool): if False, do not write over older files.
    
    Returns:
        dict: a Doc-UFCN dictionary of the form::

            { "img_size": [ <img_width>, <img_height> ],
              " textline": [
                    { "confidence": 1.0, "polygon": [[<x1,y1>], [<x2,y2>], ..., [<xn,yn>]] },
                    ...
                ]
            }
    """
    segmentation_dict = {}
    if type( segmentation_json ) is dict:
        segmentation_dict = segmentation_json
    else:
        with open( segmentation_json, 'r' ) as json_file:
            segmentation_dict = json.load( json_file )

    new_segdict = {
            "img_size": [ segmentation_dict["image_width"], segmentation_dict["image_height"]],
            "textline": [],
    }
    for line in segmentation_dict["regions"][0]["lines"]:
        new_segdict["textline"].append({
            "confidence": 1.0,
            "polygon": line['coords'],
            })

    if output_file_path and overwrite_existing:
        with open( output_file_path, 'w') as output_file:
            output_file.write( json.dumps( new_segdict, indent=2 ))
    return new_segdict

def docufcn_label_to_didip_json( segmentation_json: Path, segfile_suffix='.json', img_suffix='.img.jpg', output_file_path='', overwrite_existing=True)->dict:
    """ Convert a Doc-UFCN label file into a DiDip JSON segmentation file.
    Note: assumes a single-region file; only for evaluation use in segmentation pipeline.

    Args:
        segmentation_file (Union[Path,dict]): path to a segmentation file, Doc-UFCN style, of the form::

            { "img_size": [ <img_width>, <img_height> ],
              " textline": [
                    { "confidence": 1.0, "polygon": [[<x1,y1>], [<x2,y2>], ..., [<xn,yn>]] },
                    ...
                ]
            }

        img_suffix (str): image file suffix.
        segfile_suffix (str): _input_ segmentation file suffix.
        output_file_path (str): output file; if empty, use the standard output.
        overwrite_existing (bool): if False, do not write over older files.

    Returns:
        dict: a loose, ad-hoc DiDip-style JSON segmentation dictionary, that is not meant to be validated.
    """
    with open(segmentation_json) as seg_if:
        segdict = json.load( seg_if )
        image_width, image_height = segdict['img_size']
        new_segdict = {
                "metadata": {
                    "created": datetime.now().isoformat("T","seconds"),
                    "creator": __name__,
                    "comment": "Image name is a guess!",
                },
                "image_filename": segmentation_json.name.replace( segfile_suffix, img_suffix ),
                "image_width": image_width,
                "image_height": image_height,
                "regions": [
                    { 
                        "id": "r0",
                        "coords": [ [0,0], [image_width-1, 0], [image_width-1, image_height-1], [0, image_height-1]],
                        "lines": [ { "id": f"l{i}", "coords": l['polygon']  } 
                                    for i,l in enumerate(segdict['textline']) 
                                ],
                        
                    }
                ]
        }
        if output_file_path and overwrite_existing:
            with open( output_file_path, 'w') as output_file:
                output_file.write( json.dumps( new_segdict, indent=2 ))
        return new_segdict


def line_binary_mask_stack_from_json_file( segmentation_json: str ) -> Tensor:
    """From a JSON file describing polygons, return a stack of boolean masks where any pixel belonging
    to a polygon is 1 and the other pixels 0.

    Args:
        segmentation_json (str): a JSON file describing the lines.

    Returns:
        Tensor: a boolean tensor with size (N,H,W)
    """
    with open( segmentation_json, 'r' ) as json_file:
        return line_binary_mask_stack_from_segmentation_dict( json.load( json_file ))


def line_binary_mask_stack_from_segmentation_dict( segmentation_dict: dict ) -> Tensor:
    """From a segmentation dictionary describing polygons, return a stack of boolean masks where any pixel belonging
    to a polygon is 1 and the other pixels 0.

    Args:
        segmentation_dict (dict): a dictionary, typically constructed from a JSON file.

    Returns:
        Tensor: a boolean tensor with size (N,H,W)
    """
    polygon_boundaries = line_polygons_from_segmentation_dict( segmentation_dict)
    # create 2D boolean matrix
    mask_size = (segmentation_dict['image_width'], segmentation_dict['image_height'])
    return torch.tensor( np.stack( [ ski.draw.polygon2mask( mask_size, polyg ).transpose(1,0) for polyg in polygon_boundaries ]))


def line_polygons_from_segmentation_dict( segmentation_dict: dict, factor=1.0 ) -> list[list[int]]:
    """From a segmentation dictionary describing polygons, return a list of polygon boundaries, i.e. lists of points.

    Args:
        segmentation_dict (dict): a dictionary, typically constructed from a JSON file. The 'lines' entry is either
            top-level key, or nested as in 'regions > region > lists'.
        factor (float): the factor applied to the strip's height; if 1.0, the polygons are extracted as they are
            stored; otherwise, a new polygon is constructed from the baseline and the scaled height.

    Returns:
        list[list[int]]: a list of lists of coordinates.
    """
    flat_dict = sgf.flatten_segmentation_dict( segmentation_dict )
    if factor==1.0:
        return [ l['coords'] for l in flat_dict['lines'] ]
    line_polygons = []
    id_to_reg = { r['id']:r for r in flat_dict['regions'] }
    for line in flat_dict['lines']:
        # look for innermost containing region
        ltrb = tuple(np.array( id_to_reg[line['parents'][0]]['coords'])[[0,2]].flatten())
        line_polygons.append( polygon_utils.strip_from_baseline( line['baseline'], line['x-height'], factor, ltrb=ltrb ) if 'x-height' in line else line['coords'] )
    return line_polygons
 

def line_metrics_from_segmentation_dict( segmentation_dict: dict) -> dict:
    """From a segmentation dictionary, return basic line metrics.

    Args:
        segmentation_dict (dict): a dictionary, typically constructed from a JSON file. The 'lines' entry is either
        top-level key, or nested as in 'regions > region > lists'.
    Returns:
        dict: a list of dictionary.
    """
    lines = [ ld for ld in sgf.line_dicts_from_segmentation_dict( segmentation_dict ) if len(ld['baseline'])>=2 ]
    x_heights = np.array([ l['x-height'] for l in lines ])
    line_spacings = -1
    if len(lines)>=3:
        # subtract means of baseline's y-values 
        line_spacings = [ np.abs(np.mean([ pt[1] for pt in lines[l]['baseline']])-np.mean([ pt[1] for pt in lines[l+1]['baseline']])) for l in range(len(lines)-1) ]

    metrics_dict = { 
             'x_height_avg': np.mean( x_heights),
             'x_height_std': np.var( x_heights ),
             'line_spacing_avg': np.mean( line_spacings),
             'line_spacing_std': np.std( line_spacings),
            }
    return { k:v.round().item() for k,v in metrics_dict.items() }


def line_images_from_img_page_xml_files(img: str, page_xml: str, as_dictionary=False ) -> list[tuple[np.ndarray, np.ndarray]]:
    """From an image file path and a segmentation PageXML file describing polygons, return
    a list of pairs (<line cropped BB>, <polygon mask>), or optionally a full page dictionary with
    those enriched lines as a top element.

    Args:
        img (str): the input image's file path
        page_xml (str): a Page XML file describing the lines.
        as_dictionary (bool): return segmentation dict where each line is a tuple (<img>,<msk>,<line_dict>); useful
            for keeping track of line ids when running inference.

    Returns:
        list: a list of pairs (<line image BB>: np.ndarray (HWC), mask: np.ndarray (HW)), or a page segmentation
            dictionary with 'lines' as extra, top-level element.
    """
    with Image.open(img, 'r') as img_wh:
        segmentation_dict = sgf.segmentation_dict_from_page_xml( page_xml )
        line_pairs = line_images_from_img_segmentation_dict( img_wh, segmentation_dict )
        if not as_dictionary:
            return line_pairs
        segmentation_dict['lines']=list(zip( *(zip(*line_pairs)), sgf.line_dicts_from_segmentation_dict( segmentation_dict)))

        return segmentation_dict

def line_images_from_img_alto_files(img: str, alto_xml: str, as_dictionary=False ) -> list[tuple[np.ndarray, np.ndarray]]:
    """From an image file path and a segmentation ALTO file describing polygons, return
    a list of pairs (<line cropped BB>, <polygon mask>), or optionally a full page dictionary with
    those enriched lines as a top element.

    Args:
        img (str): the input image's file path
        alto_xml (str): a ALTO XML file describing the lines.
        as_dictionary (bool): return segmentation dict where each line is a tuple (<img>,<msk>,<line_dict>); useful
            for keeping track of line ids when running inference.

    Returns:
        list: a list of pairs (<line image BB>: np.ndarray (HWC), mask: np.ndarray (HW)), or a page segmentation
            dictionary with 'lines' as extra, top-level element.
    """
    with Image.open(img, 'r') as img_wh:
        segmentation_dict = sgf.alto_to_segmentation_dict( alto_xml )
        line_pairs = line_images_from_img_segmentation_dict( img_wh, segmentation_dict )
        if not as_dictionary:
            return line_pairs
        segmentation_dict['lines']=list(zip( *(zip(*line_pairs)), sgf.line_dicts_from_segmentation_dict( segmentation_dict)))

        return segmentation_dict


def line_images_from_img_json_files( img: str, segmentation_json: str, as_dictionary=False, factor=1.0 ) -> list[tuple[np.ndarray, np.ndarray]]:
    """From an image file path and a segmentation JSON file describing polygons, return
    a list of pairs (<line cropped BB>, <polygon mask>), or optionally a full page dictionary with
    those enriched lines as a top element.

    Args:
        img (str): the input image's file path
        segmentation_json (str): path of a JSON file
        as_dictionary (bool): return segmentation dict where each line is a tuple (<img>,<msk>,<line_dict>); useful
            for keeping track of line ids when running inference.
        factor (float): scale line polygon height to <factor>.

    Returns:
        Union[list,dict]: a list of pairs (<line image BB>: np.ndarray (HWC), mask: np.ndarray (HW)), or a page segmentation
            dictionary with 'lines' as extra, top-level element.
    """
    with Image.open(img, 'r') as img_wh, open( segmentation_json, 'r' ) as json_file:
        segmentation_dict = json.load( json_file )
        line_pairs = line_images_from_img_segmentation_dict( img_wh, segmentation_dict, factor=factor )
        if not as_dictionary:
            return line_pairs
        segmentation_dict['lines']=list(zip( *(zip(*line_pairs)), sgf.line_dicts_from_segmentation_dict( segmentation_dict)))
        return segmentation_dict


def line_images_from_img_segmentation_dict(img_whc: Image.Image, segmentation_dict: dict, factor=1.0 ) -> list[tuple[np.ndarray, np.ndarray]]:
    """From a segmentation dictionary describing polygons, return 
    a list of pairs (<line cropped BB>, <polygon mask>).

    Args:
        img_whc (Image.Image): the input image (needed for the size information).
        segmentation_dict (dict) a dictionary, typically constructed from a JSON file.
        factor (float): scale line polygon height to <factor>.

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: a list of pairs (<line
        image BB>: np.ndarray (HWC), mask: np.ndarray (HWC))
    """
    polygon_boundaries = line_polygons_from_segmentation_dict( segmentation_dict, factor=factor)
    img_hwc = np.asarray( img_whc )

    pairs_line_bb_and_mask = []# [None] * len(polygon_boundaries)

    for lbl, polyg in enumerate( polygon_boundaries ):

        points = np.array( polyg )[:,::-1] # polygon's points ( x <-> y )
        page_polyg_mask = ski.draw.polygon2mask( img_hwc.shape, points ) # np.ndarray (H,W,C)
        y_min, x_min, y_max, x_max = np.min( points[:,0] ), np.min( points[:,1] ), np.max( points[:,0] ), np.max( points[:,1] )
        line_bbox = img_hwc[y_min:y_max+1, x_min:x_max+1] # crop both img and mask
        # note: mask has as many channels as the original image
        bb_label_mask_hwc = page_polyg_mask[y_min:y_max+1, x_min:x_max+1]

        #pairs_line_bb_and_mask[lbl]=( line_bbox, bb_label_mask )
        pairs_line_bb_and_mask.append( (line_bbox, bb_label_mask_hwc) )

    return pairs_line_bb_and_mask


def line_images_from_img_polygon_map(img_wh: Image.Image, polygon_map_chw: Tensor) -> list[tuple[np.ndarray, np.ndarray]]:
    """From a tensor storing polygons, return a list of pairs (<line cropped BB>, <polygon mask>).

    Args:
        img_whc (Image.Image): the input image (needed for the size information).
        segmentation_dict (dict): a dictionary, typically constructed from a JSON file.

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: a list of pairs (<line image BB>: np.ndarray (HWC), mask: np.ndarray (HW))
    """

    max_label = torch.max( polygon_map_chw )
    img_hwc = np.array( img_wh )

    pairs_line_bb_and_mask = []# [None] * max_label

    for lbl in range(1, max_label+1 ):
        page_label_mask_hw = retrieve_polygon_mask_from_map( polygon_map_chw, lbl )

        # BB of non-zero pixels
        non_zero_ys, non_zero_xs = page_label_mask_hw.numpy().nonzero()
        y_min, x_min, y_max, x_max = np.min(non_zero_ys), np.min(non_zero_xs), np.max(non_zero_ys), np.max(non_zero_xs)
        line_bbox = img_hwc[y_min:y_max+1, x_min:x_max+1]

        bb_label_mask = expand_flat_tensor_to_n_channels(page_label_mask_hw[y_min:y_max+1, x_min:x_max+1], 3)

        #pairs_line_bb_and_mask[ lbl-1 ]=(line_bbox, bb_label_mask) 
        pairs_line_bb_and_mask.append( (line_bbox, bb_label_mask) )

    return pairs_line_bb_and_mask


def line_masks_from_img_page_xml_files(img: str, page_xml: str ) -> list[tuple[np.ndarray, np.ndarray]]:
    """From an image file path and a segmentation PageXML file describing polygons, return
    the bounding box coordinates and the boolean masks.

    Args:
        img (str): the input image's file path
        page_xml (page_xml): str a Page XML file describing the lines.

    Returns:
        tuple[np.ndarray,np.ndarray]: a pair of tensors: a tensor (N,4) of BB coordinates tuples,
            and a tensor (N,H,W) of page-wide line masks.
    """
    with Image.open(img, 'r') as img_wh:
        segmentation_dict = sgf.segmentation_dict_from_page_xml( page_xml )
        return line_masks_from_img_segmentation_dict( img_wh, segmentation_dict )


def line_masks_from_img_json_files( img: str, segmentation_json: str, key='coords' ) -> list[tuple[np.ndarray, np.ndarray]]:
    """From an image file path and a segmentation JSON file describing polygons, return
    the bounding box coordinates and the boolean masks.

    Args:
        img (str): the input image's file path
        segmentation_json (str): path of a JSON file

    Returns:
        tuple[np.ndarray,np.ndarray]: a pair of tensors: a tensor (N,4) of BB coordinates tuples,
            and a tensor (N,H,W) of page-wide line masks.
    """
    with Image.open(img, 'r') as img_wh, open( segmentation_json, 'r' ) as json_file:
        return line_masks_from_img_segmentation_dict( img_wh, json.load( json_file ), key=key)


def line_masks_from_img_segmentation_dict(img_whc: Image.Image, segmentation_dict: dict ) -> list[tuple[np.ndarray, np.ndarray]]:
    """From a segmentation dictionary describing polygons, return 
    the bounding box coordinates and the boolean masks.

    Args:
        img_whc (Image.Image): the input image (needed for the size information).
        segmentation_dict: :type segmentation_dict: dict a dictionary, typically constructed from a JSON file.

    Returns:
        tuple[np.ndarray,np.ndarray]: a pair of tensors: a tensor (N,4) of BB coordinates tuples,
            and a tensor (N,H,W) of page-wide line masks.
    """
    polygon_boundaries = line_polygons_from_segmentation_dict( segmentation_dict)

    img_hwc = np.asarray( img_whc )

    bbs = []
    masks = []

    for polyg in polygon_boundaries:
        points = np.array( polyg )[:,::-1]  # polygon's points ( x <-> y )
        page_polyg_mask = ski.util.img_as_ubyte(ski.draw.polygon2mask( img_hwc.shape[:2], points )) # np.ndarray (H,W)
        y_min, x_min, y_max, x_max = [ float(p) for p in (np.min( points[:,0] ), np.min( points[:,1] ), np.max( points[:,0] ), np.max( points[:,1] )) ]
        bbs.append( (x_min,y_min,x_max,y_max) )
        masks.append( page_polyg_mask )

    return (np.stack( bbs ), np.stack( masks ))


def expand_flat_tensor_to_n_channels( t_hw: Tensor, n: int ) -> np.ndarray:
    """Expand a flat map by duplicating its only channel into n identical ones.
    Channels dimension is last for convenient use with PIL images.

    Args:
        t_hw (Tensor): a flat map.
        n (int): number of (identical) channels in the resulting tensor.

    Returns:
        np.ndarray: a (H,W,n) array.
    """
    if len(t_hw.shape) != 2:
        raise TypeError("Function expects a 2D map!")
    t_hwc = t_hw.reshape( t_hw.shape+(1,)).expand(-1,-1,n)
    return t_hwc.numpy()

def crops_from_segdict( img: Image.Image, segdict: dict, force_rgb=False, ignore_empty_regions=False ):
    """From a segmentation dictionary, return the text regions and their
    corresponding image crops (nested regions are ignored).

    Args:
        img (Image.Image): Image to crop.
        segdict (dict): a segmentatino dictionary
        force_rgb (bool): convert binary/gray images to RGB (default: False).
        ignore_empty_regions (bool): ignore those regions that do not have any lines: useful
            when re-segmenting with inherited PageXML as layout files (default: False).
    Returns:
        tuple[list[Image.Image], list[str]]: a tuple with
            - a list of images (HWC)
            - a list of box coordinates (LTRB)
    """
    if 'regions' not in segdict:
        return tuple()
    # make it easier to check for empty regions
    if force_rgb and img.mode != 'RGB':
        img = img.convert('RGB')
    return tuple( zip( *[ ( img.crop( tuple(r['coords'][0]+r['coords'][2])), r['coords'][0]+r['coords'][2], None) for r in segdict['regions'] if (not ignore_empty_regions or ('lines' in r and len(r['lines']))) ]) )


def layout_regseg_to_crops( img: Image.Image, regseg: dict, region_labels: list[str], force_rgb=False ) -> tuple[list[Image.Image], list[str]]:
    """From a layout-app segmentation dictionary, return the regions with matching
    labels as a list of images.

    Args:
        img (Image.Image): Image to crop.
        regseg (dict): the regional segmentation json, as given by the 'layout' app
        region_labels (list[str]): Labels to be extracted.
        force_rgb (bool): convert binary/gray images to RGB (default: False).

    Returns:
        tuple[list[Image.Image], list[str]]: a tuple with 
            - a list of images (HWC)
            - a list of box coordinates (LTRB)
            - a list of class names
    """
    if 'class_names' in regseg:
        clsid_2_clsname = { i:n for (i,n) in enumerate( regseg['class_names'] )}
        to_keep = [ i for (i,v) in enumerate( regseg['rect_classes'] ) if clsid_2_clsname[v] in region_labels ]

        if force_rgb and img.mode != 'RGB':
            img = img.convert('RGB')
        return tuple( zip(*[ ( img.crop( regseg['rect_LTRB'][i] ),
                      regseg['rect_LTRB'][i],
                      clsid_2_clsname[ regseg['rect_classes'][i]]) for i in to_keep ]))
    return tuple()


def layout_regseg_check_class(regseg: dict, region_labels: list[str] ) -> list[bool]:
    """From a layout-app segmentation dictionary, check if rectangle with given labels
    have been detected.

    Args:
        regseg (dict): the regional segmentation json, as given by the 'layout' app
        region_labels (list[str]): Labels to check.

    Returns:
        list[bool]: a list of boolean values.
    """

    clsname_2_clsid = { n:i for (i,n) in enumerate( regseg['class_names'] )}
    
    output = None
    try:
        output = [ clsname_2_clsid[l] in regseg['rect_classes'] for l in region_labels ]
    except KeyError as e:
        print(f"Class label {e} does not exist in the segmentation file.")
    return output



def tile_img( img_wh: tuple[int,int], size, constraint=20, channel_dim=2 )->list[list]:
    """ Slice an image into patches: return list of patch coordinates.

    Args:
        image size (tuple[int,int]): (width, height) of image
        size (int): size of the patch square.
        constraint (int): minimum overlap between patches.
        channel_dim (int): which dimension stores the channels: 0 or 2 (default).
    Returns:
        list[list]: a list of pairs [top,left] coordinates.
    """
    width, height = img_wh
    assert height >= size and width >= size
    x_pos, y_pos = [], []
    if width == size:
        x_pos = [0]
    else:
        col = math.ceil( width / size )
        if (col*size - width)/(col-1) < constraint:
            col += 1
        overlap = (col*size - width)//(col-1)
        x_pos = [ c*(size-overlap) if c < col-1 else width-size for c in range(col) ]
    if height == size:
        y_pos = [0]
    else:
        row = math.ceil( height / size )
        if (row*size - height)/(row-1) < constraint:
            row += 1
        overlap = (row*size - height)//(row-1)
        y_pos = [ r*(size-overlap) if r < row-1 else height-size for r in range(row) ]
    return list(itertools.product(y_pos, x_pos ))


def get_binary_mask( img_whc: Image.Image, thresholding_alg: Callable=ski.filters.threshold_otsu ) -> Tensor:
    """Compute a binary mask from an image, using the given thresholding algorithm: FG=1s, BG=0s

    Args:
        img (PIL image): input image

    Returns:
        Tensor: a binary map with FG pixels=1 and BG=0.
    """
    img_hwc= np.array( img_whc )
    threshold = thresholding_alg( ski.color.rgb2gray( img_hwc ) if img_hwc.shape[2]>1 else img_hwc )*255
    img_bin_hw = torch.tensor( (img_hwc < threshold)[:,:,0], dtype=torch.bool )

    return img_bin_hw

def promote_regions_from_segdict( segdict: dict, image_folder_path: Path, validate=False):
    """
    From a segmentation dictionary, promote regions as new stand-alone images
    and create 1+ dictionaries accordingly. Assumes that 

    + regions are top-level elements: this is normally ensured by the DiDip JSON format;
    + lines are entirely contained within their region: if needed, use this module's 
      repair routine: `segdict_reassign_lines` or `json_doctor`.

    Args:
        segdict (dict): a JSON segmentation dictionary.
        image_folder_path (Path): the image's parent folder.
        validate (bool): if True, validate resulting dictionaries against schema; default is False.

    Returns:
        list[tuple[Image,dict]]: a list of tuples (image,dictionary).
    """

    region_list = []
    for reg_idx, region in enumerate(segdict['regions']):
        new_segdict = copy.deepcopy(segdict)
        new_segdict['metadata']['created']=datetime.now().isoformat("T","seconds")
        new_segdict['regions'] = new_segdict['regions'][reg_idx:reg_idx+1]
        # new region coordinates (crop-wide)
        new_segdict['regions'][0]['coords'] = (np.array( region['coords'] ) - region['coords'][0]).tolist()
        # new image dimensions
        new_segdict['image_width'], new_segdict['image_height']= new_segdict['regions'][0]['coords'][2]
        x_offset, y_offset = region['coords'][0]
        # offset lines
        for line_idx, line in enumerate(region['lines']):
            for attr in ('coords', 'centerline', 'baseline'):
                new_coords=np.array(line[attr])-[x_offset, y_offset]
                print(new_coords)
                assert np.all( new_coords >= 0 )
                new_segdict['regions'][0]['lines'][line_idx][attr]=new_coords.tolist()
        # crop region
        with Image.open( image_folder_path.joinpath( segdict['image_filename'] )) as page_img:
            #print(np.array( region['coords'])[[0,2]].flatten())
            region_img = page_img.crop( tuple(np.array( region['coords'])[[0,2]].flatten().tolist() ))
            region_img_filename = re.sub(r'\.(img\.)?(png|jpg)$', f".r{reg_idx}"+r'\g<0>', segdict['image_filename'])
            new_segdict['image_filename']=region_img_filename
            assert region_img.size == (new_segdict['image_width'], new_segdict['image_height'])
            if validate:
                assert json_validate( new_segdict )
        region_list.append( (region_img, new_segdict) )
    return region_list



def promote_regions_from_json_file( filename: Path, validate=False ):
    """
    From a segmentation file, promote regions as new stand-alone images
    and create 1+ dictionaries accordingly. Assumes that 

    + regions are top-level elements: this is normally ensured by the DiDip JSON format;
    + lines are entirely contained within their region: if needed, use this module's 
      repair routine: `segdict_reassign_lines` or `json_doctor`.

    Args:
        filename (Union[str,Path]): a JSON segmentation file.
        validate (bool): if True, validate resulting dictionaries against schema; default is False.

    Returns:
        list[tuple[Image,dict]]: a list of tuples (image,dictionary).
    """
    parent_path = Path(filename).parent
    with open( filename, 'r') as json_if:
        segdict = json.load( json_if )
        image_file = parent_path.joinpath( segdict['image_filename'])
        if not image_file.exists():
            print("Could not find image {image_file}. Abort.")
            return []
        return promote_regions_from_segdict( segdict, parent_path, validate=validate )

