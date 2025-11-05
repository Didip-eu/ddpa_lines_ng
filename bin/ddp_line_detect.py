#!/usr/bin/env python3

# nprenet@gmail.com
# 05.2025

"""
Line detection app, for:

    - page-wide line detection, with no consideration for regions
    - region-based line detection, provided a class name (ex. Wr:Oldtext) and the existence of layout segmentation file for each image

The heavy lifting is done by the associated Mask-RCNN appn that computes morphological features for each line map. This app deals with what comes in, and out.

Output formats: 
    + PageXML: core polygon and baseline only.
    + JSON: custom format, including features that are in the PageXML spec: centerline, line height, extended boundaries.
    + npy (2D-label map only)

Example calls::
    
    export FSDB_ROOT=~/tmp/data/1000CV
    PYTHONPATH=. python3 ./bin/ddp_line_detect -img_paths "${FSDB_ROOT}"/*/*/d9ae9ea49832ed79a2238c2d87cd0765/*layout.crops/*OldText*.jpg -model_path best.mlmodel -region_classes Wr:OldText

    # patch-trained model, exporting raw polygons (instead of abstract reconstructions)
    PYTHONPATH=. python3 ./bin/ddp_line_detect -img_paths "${FSDB_ROOT}"/*/*/d9ae9ea49832ed79a2238c2d87cd0765/*layout.crops/*OldText*.jpg -model_path best.mlmodel -region_classes Wr:OldText -raw_polygons 1

TODO:
"""
# stdlib
import sys
from pathlib import Path
import json
import re
import sys
from datetime import datetime
import logging
import itertools
import shutil
import math
import gzip
from hashlib import md5
from time import time

# 3rd party
from PIL import Image
import skimage as ski
import numpy as np

# Didip
import fargv

# local

src_root = Path(__file__).parents[1]
sys.path.append( str( src_root ))
from libs import seglib, list_utils as lu, line_geometry as lgm
from bin import ddp_lineseg_train as lsg


logging.basicConfig( level=logging.DEBUG, format="%(asctime)s - %(levelname)s: %(funcName)s - %(message)s", force=True )
logger = logging.getLogger(__name__)

# tone down unwanted logging
logging.getLogger('PIL').setLevel(logging.INFO)


p = {
        "appname": "lines",
        "model_path": str(src_root.joinpath("best.mlmodel")),
        #"img_paths": set([Path.home().joinpath("tmp/data/1000CV/AT-AES/d3a416ef7813f88859c305fb83b20b5b/207cd526e08396b4255b12fa19e8e4f8/4844ee9f686008891a44821c6133694d.img.jpg")]),
        "img_paths": set([]),
        "charter_dirs": set([]),
        "region_classes": [set([]), "Names of the layout-app regions on which lines are to be detected. Eg. '[Wr:OldText']. If empty (default), detection is run on the entire page."],
        "img_suffix": [r".img.*p*g", "Image file suffix."],
        "layout_suffix": [".layout.pred.json", "Regions are given by segmentation file that is <img name stem><suffix>."],
        "line_attributes": [set(["centerline", "height"]), "Non-standard line properties to be included in the dictionary."],
        "output_format": [("json", "xml", "npy", "stdout"), "Segmentation output: json=<JSON file>, xml=<PageXML file>, npy=label map (HW), stdout=JSON on standard output."],
        "output_dir": ['', "Output directory; if not provided, defaults to the image path's parent."],
        'mask_threshold': [.6, "In the post-processing phase, threshold to use for line soft masks."],
        'box_threshold': [0.75, "Threshold used for line bounding boxes."],
        'patch_row_count': [ 0, "Process the image in <patch_row_count> rows."],
        'patch_col_count': [ 0, "Process the image in <patch_col_count> cols."],
        'patch_size': [1024, "Process the image by <patch_size>*<patch_size> patches"],
        'cache_predictions': [0, "Cache prediction tensors for faster, repeated calls with various post-processing options."],
        'cached_prediction_root_dir': ['/tmp', "Where to save the cached predictions."],
        'raw_polygons': [1, "Show polygons as resulting from the NN (default); otherwise, show the abstract polygons constructed from centerlines."],
        'line_height_factor': [1.0, "Factor (within ]0,1]) to be applied to the polygon height: allows for extracting polygons that extend above and below the core line-unused if 'raw_polygons' set"],
        'overwrite_existing': [1, "Write over existing output file (default)."],
        'timer': [0, "Aggregate performance metrics. A strictly positive integer <n> computes the mean time for every batch of <n> images."],
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


def build_segdict( img_metadata, segmentation_record, line_attributes, contour_tolerance=4.0 ):
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
    segdict = { 'created': str(datetime.now()), 'creator': __file__, }
    segdict.update( img_metadata )
    segdict['regions']=[ { 'id': 'r0', 'type': 'text_region', 'coords': [], 'lines': [] } ]

    mp, atts = segmentation_record
    line_id=0
    for att_dict in atts:
        label, centroid, polygon_coords, line_height, centerline, baseline = [ att_dict[k] for k in ('label','centroid', 'polygon_coords', 'line_height', 'centerline', 'baseline')]
        logger.debug("polygon_coords.shape={}, polygon_coords".format(polygon_coords.shape))
        dict_line_entry = {'id': f'l{line_id}', 'coords': polygon_coords[:,::-1].tolist(), 'baseline': baseline[:,::-1].tolist() }
        if 'height' in line_attributes:
            dict_line_entry['height']=int(line_height)
        if 'centerline' in line_attributes:
            dict_line_entry['centerline']=centerline[:,::-1].tolist() # yx to xy
        if 'baseline' in line_attributes:
            dict_line_entry['baseline']=baseline[:,::-1].tolist()
        segdict['regions'][0]['lines'].append( dict_line_entry )
        line_id += 1
    # boundary of the dummy region
    all_points = np.array(list(itertools.chain.from_iterable([ l['coords'] for reg in segdict['regions'] for l in reg['lines'] ])))
    l, t, r, b = [ int(p) for p in ( min( all_points[:,0]), min( all_points[:,1]), max( all_points[:,0]), max( all_points[:,1]))]
    segdict['regions'][-1]['coords']=[[l,t],[r,t],[r,b],[l,b]]
    return segdict


def build_segdict_composite( img_metadata, boxes, segmentation_records, line_attributes, contour_tolerance=4.0):
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
    segdict = { 'created': str(datetime.now()), 'creator': __file__, }
    segdict.update( img_metadata )
    segdict['regions']=[]

    region_id = 0
    for box, record in zip(boxes, segmentation_records):
        this_region_lines = []
        line_id = 0
        _, atts = record
        offset = np.array([box[1],box[0]])
        for att_dict in atts:
            label, polygon_coords, line_height, centerline, baseline = [ att_dict[k] for k in ('label','polygon_coords','line_height', 'centerline', 'baseline')]
            # adding box offsets
            polygon_coords += offset.astype(polygon_coords.dtype)
            baseline += offset.astype(baseline.dtype)
            centerline += offset.astype(centerline.dtype)
            dict_line_entry = {'id': f'l{line_id}', 'coords': polygon_coords[:,::-1].tolist(), 'baseline': baseline[:,::-1].tolist() }
            if 'height' in line_attributes:
                dict_line_entry['height']=int(line_height)
            if 'centerline' in line_attributes:
                dict_line_entry['centerline']=centerline[:,::-1].tolist() # yx to xy
            if 'baseline' in line_attributes:
                dict_line_entry['baseline']=baseline[:,::-1].tolist()
            this_region_lines.append( dict_line_entry )
            line_id += 1
        segdict['regions'].append( { 'id': f'r{region_id}', 'type': 'text_region', 'coords': [[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]], 'lines': this_region_lines } )
        region_id += 1

    return segdict


def check_patch_size_against_model( live_model: dict, patch_size ):
    if 'train_style' in live_model.hyper_parameters:
        if live_model.hyper_parameters['train_style'] != 'patch':
           logger.warning('The model being loaded was _not_ trained on fixed-size patches: expect suboptimal results.')
        elif live_model.hyper_parameters['img_size'][0] != args.patch_size:
           logger.warning('The model being loaded is trained on {}x{} patches, but the script uses a {} patch size argument: overriding patch_size value with model-stored size.'.format( *live_model.hyper_parameters['img_size'], args.patch_size))
           return live_model.hyper_parameters['img_size'][0]
    return patch_size


def pack_img_inputs_outputs( args:dict ):
    """
    Process arguments into a tuple of the form. It is a triplet::

        ( <img path>, <layout segmentation path>, <line segmentation path> ) 

    No existence check on the layout segmentation path.
    """
    all_img_paths = set([ Path(p) for p in args.img_paths ])

    for charter_dir in args.charter_dirs:
        charter_dir_path = Path( charter_dir )
        if charter_dir_path.is_dir() and charter_dir_path.joinpath("CH.cei.xml").exists():
            new_imgs = charter_dir_path.glob("*{}".format(args.img_suffix))
            all_img_paths = all_img_paths.union( charter_dir_path.glob("*{}".format(args.img_suffix)))
    path_triplets = []
    for img_path in all_img_paths:
        img_stem = re.sub(r'{}$'.format( args.img_suffix), '', img_path.name )
        layout_segfile_path = Path( re.sub(r'{}$'.format( args.img_suffix), args.layout_suffix, str(img_path) ))
        output_dir = img_path.parent if not args.output_dir else Path(args.output_dir)
        path_triplets.append( ( img_path, layout_segfile_path, output_dir.joinpath( f'{img_stem}.{args.appname}.pred.{args.output_format}')))
    #return path_triplets
    return sorted( path_triplets, key=lambda x: str(x))


if __name__ == "__main__":

    args, _ = fargv.fargv( p )

    cached_prediction_root_path = Path(args.cached_prediction_root_dir)
    cached_prediction_subdir_path = None
    cache_subdir_path = None
    with open( args.model_path, 'rb') as mf:
        # computing MD5 of model file used
        model_md5 = md5( mf.read() ).hexdigest()
        # create output subdir for this model
        cached_prediction_subdir_path = cached_prediction_root_path.joinpath( model_md5 )
        cached_prediction_subdir_path.mkdir( exist_ok=True )
        model_local_copy_path = cached_prediction_subdir_path.with_suffix('.mlmodel')
        # copy model file into root folder, with MD5 identifier (make it easier to rerun eval loops later)
        if not model_local_copy_path.exists():
            shutil.copy2( args.model_path, model_local_copy_path )
        if args.cache_predictions:
            cache_subdir_path = cached_prediction_subdir_path.joinpath('cached') 
            cache_subdir_path.mkdir( exist_ok=True )
            logger.info( 'Using cache subdirectory {}.'.format( cache_subdir_path ))

    if not Path( args.model_path ).exists():
        raise FileNotFoundError("Could not find model file", args.model_path)
    live_model = lsg.SegModel.load( args.model_path ) 

    if args.raw_polygons and args.line_height_factor != 1.0:
        logger.warning("'-raw_polygons' option set: ignoring the line height factor ({}).".format( args.line_height_factor))

    # Store aggregate computation time for every batch of 100 images 
    timer_means = []
    start_time = time()

    for img_idx, img_triplet in enumerate( pack_img_inputs_outputs( args )): 
        img_path, layout_file_path, output_file_path = img_triplet
        logger.info( "File triplet={}".format( img_triplet ))
        
        img_md5=''
        if args.cache_predictions:
            with open(img_path, 'rb') as imgf:
                img_md5 = md5( imgf.read()).hexdigest()

        with Image.open( img_path, 'r' ) as img:
            # keys from PageXML specs
            img_metadata = { 'image_filename': str(img_path.name), 'image_width': img.size[0], 'image_height': img.size[1] }

            binary_mask, segdict = None, {}

            # Option 1: segment the region crops (from layout), and construct a page-wide file
            if len(args.region_classes):
                logger.debug(f"Run segmentation on masked regions '{args.region_classes} from layout file {layout_file_path}, instead of whole page.")
                if not layout_file_path.exists():
                    #logger.warning("Could not find layout segmentation file {}. Skipping item.".format( layout_file_path ))
                    continue
                
                # parse segmentation file, and extract and concatenate the WritableArea crops
                logger.debug("layout_file_path={}".format(layout_file_path))
                with open(layout_file_path, 'r') as regseg_if:
                    regseg = json.load( regseg_if )
                    # iterate over layout crops and segment
                    layout_data = seglib.layout_regseg_to_crops( img, regseg, args.region_classes )
                    if len(list(layout_data))==0:
                        #logger.warning("Could not find region with name in {} in the layout segmentation file {}. Skipping item.".format( args.region_classes, layout_file_path ))
                        continue
                    crops_pil, boxes, classes = seglib.layout_regseg_to_crops( img, regseg, args.region_classes, force_rgb=True )

                    binary_masks = []
                    for crop_idx, crop_whc in enumerate(crops_pil):
                        binary_mask = None

                        # Style 1: inference from fixed-size patches
                        if args.patch_size:
                            patch_size = check_patch_size_against_model( live_model, args.patch_size )
                            binary_mask = lgm.binary_mask_from_fixed_patches( crop_whc, patch_size=patch_size, model=live_model, mask_threshold=args.mask_threshold, box_threshold=args.box_threshold, cached_prediction_prefix=img_md5, cached_prediction_path=cache_subdir_path )

                        # Check for unusual size before choosing patch-based inference or not
                        else:
                            rows, cols = is_unusual_size( crop_whc.size[1], crop_whc.size[0] ) 
                            if rows or cols:
                                logger.debug("Unusual size detected: process {}x{} patches.".format(rows, cols))
                            if rows > 1:
                                binary_mask = lgm.binary_mask_from_patches( crop_whc, row_count=rows, model=model, box_threshold=args.box_threshold, mask_threshold=args.mask_threshold) 
                            elif cols > 1:
                                binary_mask = lgm.binary_mask_from_patches( crop_whc, col_count=cols, model=model, box_threshold=args.box_threshold, mask_threshold=args.mask_threshold) 
                            else:
                                imgs_t, preds, sizes = lsg.predict( [crop_whc], live_model=live_model )
                                binary_mask = lgm.post_process( preds[0], orig_size=sizes[0], box_threshold=args.box_threshold, mask_threshold=args.mask_threshold ) 
                        if binary_mask is None:
                            logger.warning("No line mask found for {}, crop {}: skipping item.".format( img_path, crop_idx ))
                            continue
                        binary_masks.append( binary_mask )

                    try:
                        segmentation_records = [ lgm.get_morphology( msk, raw_polygons=args.raw_polygons, height_factor=args.line_height_factor ) for msk in binary_masks ]
                        #binary_mask = np.squeeze( segmentation_records[0][0] )
                        segdict = build_segdict_composite( img_metadata, boxes, segmentation_records, args.line_attributes ) 
                    except (TypeError, ValueError):
                        logger.error("Failed to polygonize the line masks: abort segmentation.")
                        continue

            # Option 2: single-file segmentation (an Wr:OldText crop, supposedly)
            else:
                binary_mask = None
                if args.patch_size:
                    # case 1: process image in fixed-size patches
                    patch_size = check_patch_size_against_model( live_model, args.patch_size )
                    logger.debug('Patch size: {} x {}'.format( patch_size, patch_size))
                    binary_mask = lgm.binary_mask_from_fixed_patches( img, patch_size=patch_size, model=live_model, box_threshold=args.box_threshold, mask_threshold=args.mask_threshold, cached_prediction_prefix=img_md5, cached_prediction_path=cache_subdir_path )
                else:
                    # case 2: process image in M x N patches
                    if args.patch_row_count and args.patch_col_count:
                        logger.debug("Process {}x{} patches.".format(args.patch_row_count, args.patch_col_count))
                        binary_mask = lgm.binary_mask_from_patches( img, args.patch_row_count, args.patch_col_count, model=model, box_threshold=args.box_threshold, mask_threshold=args.mask_threshold )
                    # case 3: process image as-is
                    else:
                        logger.debug("Page-wide processing")
                        imgs_t, preds, sizes = lsg.predict( [img], live_model=live_model )
                        binary_mask = lgm.post_process( preds[0], orig_size=sizes[0], box_threshold=args.box_threshold, mask_threshold=args.mask_threshold )
                segmentation_record = lgm.get_morphology( binary_mask, raw_polygons=args.raw_polygons, height_factor=args.line_height_factor )
                segdict = build_segdict( img_metadata, segmentation_record, args.line_attributes )

            ############ 3. Handing the output #################
            logger.debug(f"Serializing segmentation for img.shape={img.size}")

            if args.output_format == 'stdout':
                print(json.dumps(segdict))
                sys.exit()
            if not output_file_path.exists() or args.overwrite_existing:
                if args.output_format == 'json':
                    with open(output_file_path, 'w') as of:
                        #segdict['image_wh']=img.size
                        of.write(json.dumps( segdict, indent=4 ))
                elif args.output_format == 'xml':
                    #segdict['image_wh']=img.size
                    seglib.xml_from_segmentation_dict( segdict, pagexml_filename=output_file_path )
                elif args.output_format == 'npy':
                    np.save( output_file_path, binary_mask )
                logger.debug("Segmentation output saved in {}".format( output_file_path ))

        if args.timer > 0 and img_idx > 0 and img_idx % args.timer==0:
            timer_means.append( (time()-start_time)/args.timer )
            start_time = time()
            logger.info("{}: {}s/img".format( img_idx, timer_means[-1]))
