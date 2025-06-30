#!/usr/bin/env python3

# nprenet@gmail.com
# 05.2025

"""
Line detection app, that can do either:

    - page-wide line detection, with no consideration for regions
    - region-based line detection, provided a class name (ex. Wr:Oldtext) and the existence of layout segmentation file for each image

The heavy lifting is done by a Mask-RCNN model that computes morphological features for each line map: this script uses them to write a JSON segmentation file.

Output formats: 
    + PageXML: core polygon and baseline only.
    + JSON: custom format, including features that are in the PageXML spec: centerline, line height, extended boundaries.
    + npy (2D-label map only)

Example call::
    export DIDIP_ROOT=. FSDB_ROOT=~/tmp/data/1000CV
    PYTHONPATH=${DIDIP_ROOT} python3 ./bin/ddp_line_detect -img_paths "${FSDB_ROOT}"/*/*/d9ae9ea49832ed79a2238c2d87cd0765/*seals.crops/*OldText*.jpg -model_path best.mlmodel -mask_classes Wr:OldText

TODO:
"""
# stdlib
import sys
from pathlib import Path
import json
import re
import sys
import datetime
import logging
import itertools
import math

# 3rd party
import torch
from PIL import Image
import skimage as ski
import numpy as np
import matplotlib.pyplot as plt

# Didip
import fargv

# local
src_root = Path(__file__).parents[1]
sys.path.append( str( src_root ))
import ddp_lineseg as lsg
from libs import seglib, list_utils as lu


logging.basicConfig( level=logging.DEBUG, format="%(asctime)s - %(levelname)s: %(funcName)s - %(message)s", force=True )
logger = logging.getLogger(__name__)

# tone down unwanted logging
logging.getLogger('PIL').setLevel(logging.INFO)



p = {
        "appname": "lines",
        "model_path": str(src_root.joinpath("best.mlmodel")),
        #"img_paths": set([Path.home().joinpath("tmp/data/1000CV/AT-AES/d3a416ef7813f88859c305fb83b20b5b/207cd526e08396b4255b12fa19e8e4f8/4844ee9f686008891a44821c6133694d.img.jpg")]),
        "img_paths": set([]),
        "charter_dirs": set(["./"]),
        "mask_classes": [set([]), "Names of the seals-app regions on which lines are to be detected. Eg. '[Wr:OldText']. If empty (default), detection is run on the entire page."],
        "region_segmentation_suffix": [".seals.pred.json", "Regions are given by segmentation file that is <img name stem>.<suffix>."],
        "centerlines": [0, "If True, compute centerlines (default is False)."],
        "output_format": [("json", "xml", "npy", "stdout"), "Segmentation output: json=<JSON file>, xml=<PageXML file>, npy=label map (HW), stdout=JSON on standard output."],
        'mask_threshold': [.25, "In the post-processing phase, threshold to use for line soft masks."],
        'patch_row_count': [ 0, "Process the image in <patch_row_count> rows."],
        'patch_col_count': [ 0, "Process the image in <patch_col_count> cols."],
}



def is_unusual_size( height, width ):
    """
    Compute number of row or col patch for images with outlier height-to-width ratio.

    Args:
        height (int): img height.
        width (int): img width.
    Returns:
        tuple(int,int): either (<rows>,1) if unusually high, (1,<cols>) if unusually wide, or (0,0)
    """
    ratio_height_to_width = height/width
    if ratio_height_to_width >= 2.0:
        return (int(math.ceil(ratio_height_to_width )), 1)
    elif ratio_height_to_width <= 0.2:
        return (1, int(math.ceil(1/ratio_height_to_width)))
    return (0,0)


def label_map_from_patches( img: Image.Image, row_count=1, col_count=1, overlap=50, model=None):
    """
    Construct a single label map from predictions on <row_count>x<col_count> patches.

    Args:
        img (Image.Image): a PIL image.
        row_count (int): number of rows.
        col_count (int): number of cols.
        overlap (int): overlap between patches (in pixels)

    Returns:
        np.ndarray: a (1,H,W) label map.
    """
    assert model is not None
    row_cuts_exact, col_cuts_exact  = [ list(int(f) for f in np.linspace(0, dim, d)) for dim, d in ((img.height, row_count+1), (img.width, col_count+1)) ]
    row_cuts, col_cuts = [[[ c+overlap, c-overlap] if c and c<cuts[-1] else c for c in cuts ] for cuts in ( row_cuts_exact, col_cuts_exact ) ]
    rows, cols = [ lu.group( lu.flatten( cut ), gs=2) for cut in (row_cuts, col_cuts) ]
    crops_yyxx=[ lu.flatten(lst) for lst in itertools.product( rows, cols ) ]
    logger.debug(crops_yyxx)
    img_hwc = np.array( img )
    img_crops = [ torch.from_numpy(img_hwc[ crop[0]:crop[1], crop[2]:crop[3] ]).permute(2,0,1) for crop in crops_yyxx ]
    crops_t, crop_preds, crop_sizes = lsg.predict( img_crops, live_model=model )
    page_mask = np.zeros((crops_yyxx[-1][1],crops_yyxx[-1][3]), dtype='bool')
    for i in range(len(crops_yyxx)):
        t,b,l,r = crops_yyxx[i]
        page_mask[t:b, l:r] += lsg.post_process( crop_preds[i], orig_size=crop_sizes[i], mask_threshold=.2 )[0]
    return page_mask[None,:]


def build_segdict( img_metadata, segmentation_record, contour_tolerance=4.0 ):
    """
    Construct the line segmentation dictionary (single-region file).

    Args:
        img_metadata (dict): original image's metadata.
        segmentation_record (tuple[np.ndarray, list[tuple]]): a tuple with
            - label map (np.ndarray)
            - a list of line attribute dicts (label, centroid pt, ..., area, polygon_coords)
        contour_tolerance (float): value for contour approximation (default: 4)
    Return:
        dict: a segmentation dictionary
    """
    segdict = { 'created': str(datetime.datetime.now()), 'creator': __file__, }
    segdict.update( img_metadata )
    segdict['regions']=[ { 'id': 'r0', 'type': 'text_region', 'boundary': [], 'lines': [] } ]

    mp, atts = segmentation_record
    line_id=0
    for att_dict in atts:
        label, polygon_coords, area, line_height, centerline = [ att_dict[k] for k in ('label','polygon_coords','area', 'line_height', 'centerline')]
        centerline = ski.measure.approximate_polygon( centerline[:,::-1], tolerance=contour_tolerance) if len(centerline) else np.array([])
        baseline = np.stack( [centerline[:,0], centerline[:,1]+int(line_height/2)], axis=1) if len(centerline) else np.array([])
        segdict['regions'][0]['lines'].append({ 
                'id': f'l{line_id}', 
                'boundary': ski.measure.approximate_polygon( polygon_coords[:,::-1], tolerance=contour_tolerance).tolist(),
                'height': int(line_height),
                'centerline': centerline.tolist(),
                'baseline': baseline.tolist(),
                })
        line_id += 1
    # boundary of the dummy region
    all_points = np.array(list(itertools.chain.from_iterable([ l['boundary'] for reg in segdict['regions'] for l in reg['lines'] ])))
    l, t, r, b = [ int(p) for p in ( min( all_points[:,0]), min( all_points[:,1]), max( all_points[:,0]), max( all_points[:,1]))]
    segdict['regions'][-1]['boundary']=[[l,t],[r,t],[r,b],[l,b]]
    return segdict


def build_segdict_composite( img_metadata, boxes, segmentation_records, contour_tolerance=4.0):
    """
    Construct the region + line segmentation dictionary.

    Args:
        img_metadata (dict): original image's metadata.
        boxes (list[tuple]): list of LTRB coordinate vectors, one for each region.
        segmentation_records (list[tuple[np.ndarray, list[tuple]]]): a list of N tuples (one
        per region) with
            - label map (np.ndarray)
            - a list of line attribute dicts (label, centroid pt, ..., area, polygon_coords)
        contour_tolerance (float): value for contour approximation (default: 4)

    Return:
        dict: a segmentation dictionary
    """
    segdict = { 'created': str(datetime.datetime.now()), 'creator': __file__, }
    segdict.update( img_metadata )
    segdict['regions']=[]

    region_id = 0
    for box, record in zip(boxes, segmentation_records):
        this_region_lines = []
        line_id = 0
        _, atts = record
        for att_dict in atts:
            label, polygon_coords, area, line_height, centerline = [ att_dict[k] for k in ('label','polygon_coords','area','line_height', 'centerline')]
            centerline = ski.measure.approximate_polygon( centerline[:,::-1], tolerance=contour_tolerance) if len(centerline) else np.array([])
            baseline = np.stack( [centerline[:,0], centerline[:,1]+int(line_height/2)], axis=1)
            this_region_lines.append({
                'id': f'r{region_id}l{line_id}',
                'boundary': ski.measure.approximate_polygon( polygon_coords[:,::-1] + box[:2], tolerance=contour_tolerance).tolist(),
                'height': int(line_height),
                'centerline': centerline.tolist(),
                'baseline': baseline.tolist()
            })
            line_id += 1
        segdict['regions'].append( { 'id': f'r{region_id}', 'type': 'text_region', 'boundary': [[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]], 'lines': this_region_lines } )
        region_id += 1
    


    return segdict


if __name__ == "__main__":

    args, _ = fargv.fargv( p )

    all_img_paths = list(sorted(args.img_paths))
    for charter_dir in args.charter_dirs:
        charter_dir_path = Path( charter_dir )
        logger.debug(f"Charter Dir: {charter_dir}")
        if charter_dir_path.is_dir() and charter_dir_path.joinpath("CH.cei.xml").exists():
            charter_images = [str(f) for f in charter_dir_path.glob("*.img.*")]
            all_img_paths += charter_images

        args.img_paths = list(all_img_paths)

    logger.debug( args )

    for path in list( args.img_paths ):
        logger.debug( path )
        path = Path(path)
        #stem = Path( path ).stem
        stem = re.sub(r'\..+', '', path.name )

        # only for segmentation on Seals-detected regions
        region_segfile = re.sub(r'.img.jpg', args.region_segmentation_suffix, str(path) )

        with Image.open( path, 'r' ) as img:
            # keys from PageXML specs
            img_metadata = { 'imagename': str(path.name), 'image_wh': list(img.size), }

            # ex. '1063063ceab07a6b9f146c598810529d.lines.pred'
            output_file_path_wo_suffix = path.parent.joinpath( f'{stem}.{args.appname}.pred' )

            json_file_path = Path(f'{output_file_path_wo_suffix}.json')
            npy_file_path = Path(f'{output_file_path_wo_suffix}.npy')

            if not Path( args.model_path ).exists():
                raise FileNotFoundError("Could not find model file", args.model_path)
            model = lsg.SegModel.load( args.model_path )
            label_map = None
            segdict = {}

            # Option 1: segment the region crops (from seals), and construct a page-wide file
            if len(args.mask_classes):
                logger.debug(f"Run segmentation on masked regions '{args.mask_classes}', instead of whole page.")
                # parse segmentation file, and extract and concatenate the WritableArea crops
                with open(region_segfile) as regseg_if:
                    regseg = json.load( regseg_if )
                   
                    # iterate over seals crops and segment
                    crops_pil, boxes, classes = seglib.seals_regseg_to_crops( img, regseg, args.mask_classes )

                    label_masks = []
                    for crop_whc in crops_pil:
                        logger.debug("Crop size={})".format(crop_whc.size))

                        rows, cols = is_unusual_size( crop_whc.size[1], crop_whc.size[0] ) 
                        if rows or cols:
                            logger.debug("Unusual size detected: process {}x{} patches.".format(rows, cols))
                        if rows > 1:
                            label_masks.append( label_map_from_patches( crop_whc, row_count=rows, model=model) )
                        elif cols > 1:
                            label_masks.append( label_map_from_patches( crop_whc, col_count=cols, model=model) )
                        else:
                            imgs_t, preds, sizes = lsg.predict( [crop_whc], live_model=model )
                            label_masks.append( lsg.post_process( preds[0], orig_size=sizes[0], mask_threshold=args.mask_threshold ) )

                        # each segpage: label map, attribute, <image path or id>
                    segmentation_records = [ lsg.get_morphology( msk, centerlines=args.centerlines) for msk in label_masks ]
                    #label_map = np.squeeze( segmentation_records[0][0] )
                    segdict = build_segdict_composite( img_metadata, boxes, segmentation_records ) 

            # Option 2: single-file segmentation (an Wr:OldText crop, supposedly)
            else:
                logger.info("Starting segmentation")
                label_map = None
                # case 1: process image in patches
                if args.patch_row_count and args.patch_col_count:
                    logger.debug("Process {}x{} patches.".format(args.patch_row_count, args.patch_col_count))
                    label_mask = label_map_from_patches( img, args.patch_row_count, args.patch_col_count, model=model )
                # case 2: process image as-is
                else:
                    logger.debug("Page-wide processing")
                    imgs_t, preds, sizes = lsg.predict( [img], live_model=model )
                    logger.info("Successful segmentation.")
                    label_mask = lsg.post_process( preds[0], orig_size=sizes[0], mask_threshold=args.mask_threshold )
                segmentation_record = lsg.get_morphology( label_mask, centerlines=args.centerlines)
                segdict = build_segdict( img_metadata, segmentation_record )

            ############ 3. Handing the output #################
            output_file_path = Path(f'{output_file_path_wo_suffix}.{args.output_format}')
            logger.debug(f"Serializing segmentation for img.shape={img.size}")

            if args.output_format == 'stdout':
                print(json.dumps(segdict))
                sys.exit()
            if args.output_format == 'json':
                with open(output_file_path, 'w') as of:
                    segdict['image_wh']=img.size
                    json.dump( segdict, of )
            elif args.output_format == 'xml':
                segdict['image_wh']=img.size
                seglib.xml_from_segmentation_dict( segdict, pagexml_filename=output_file_path )
            elif args.output_format == 'npy':
                np.save( output_file_path, label_map )
            logger.info("Segmentation output saved in {}".format( output_file_path ))


