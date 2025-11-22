#!/usr/bin/env python3

# nprenet@gmail.com
# 11.2025

"""
Line detection app, i.e. the specialized, fsdb-specific version:
+ inference is run on the regions detected by the layout app, provided a class name (ex. `Wr:Oldtext`) 
+ each region is processed in patches (default: 1024x1024), assuming that the model used has been trained accordingly

For more options (page-wide inference, flexible row/col processing, NN prediction cache), see its fully-featured companion `ddp_line_detect_full.py`.
In both apps, the heavy lifting is done by the Mask-RCNN model defined in module `ddp_lineseg_train`.

Output formats: 
+ PageXML: core polygon and baseline only.
+ JSON: custom format, including features that are not in the PageXML spec: centerline, line height.

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
from PIL import Image, UnidentifiedImageError
import skimage as ski
import numpy as np

# Didip
import fargv

# local

src_root = Path(__file__).parents[1]
sys.path.append( str( src_root ))
from libs import seglib, list_utils as lu, line_geometry as lgm
from libs.train_utils import duration_estimate
from bin import ddp_lineseg_train as lsg


logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(levelname)s: %(funcName)s - %(message)s", force=True )
logger = logging.getLogger(__name__)

# tone down unwanted logging
logging.getLogger('PIL').setLevel(logging.INFO)


p = {
        "appname": "lines",
        "model_path": str(src_root.joinpath("best.mlmodel")),
        "img_paths": set([]),
        "charter_dirs": set([]),
        "region_classes": [set(["Wr:OldText"]), "Names of the layout-app regions on which lines are to be detected. Eg. '[Wr:OldText']. If empty (default), detection is run on the entire page."],
        "img_suffix": [r".img.*p*g", "Image file suffix."],
        "layout_suffix": [".layout.pred.json", "Regions are given by segmentation file that is <img name stem><suffix>."],
        "line_attributes": [set(["centerline", "height"]), "Non-standard line properties to be included in the dictionary."],
        "output_format": [("json", "xml", "stdout", "quiet"), "Segmentation output: json=<JSON file>, xml=<PageXML file>, stdout=JSON on standard output, quiet=nothing (for testing and timing)"],
        "output_dir": ['', "Output directory; if not provided, defaults to the image path's parent."],
        'mask_threshold': [.6, "In the post-processing phase, threshold to use for line soft masks."],
        'box_threshold': [0.75, "Threshold used for line bounding boxes."],
        'patch_size': [1024, "Process the image by <patch_size>*<patch_size> patches"],
        'raw_polygons': [0, "Serialize polygons as resulting from the NN (default); otherwise, construct the abstract polygons from centerlines."],
        'device': [('cpu','gpu'), "Computing device."],
        'line_height_factor': [1.0, "Factor (within ]0,1]) to be applied to the polygon height: allows for extracting polygons that extend above and below the core line-unused if 'raw_polygons' set"],
        'overwrite_existing': [1, "Write over existing output file (default)."],
        'timer': [0, "Aggregate performance metrics. A strictly positive integer <n> computes the mean time for every batch of <n> images."],
        'timer_logs': ['stdout', "Filename for timer logs."],
}


def check_patch_size_against_model( live_model: dict, patch_size ):
    if 'train_style' in live_model.hyper_parameters:
        if live_model.hyper_parameters['train_style'] != 'patch':
           logger.warning('The model being loaded was _not_ trained on fixed-size patches: expect suboptimal results.')
        elif live_model.hyper_parameters['img_size'][0] != args.patch_size:
           logger.warning('The model being loaded is trained on {}x{} patches, but the script uses a {} patch size argument: overriding patch_size value with model-stored size.'.format( *live_model.hyper_parameters['img_size'], args.patch_size))
           return live_model.hyper_parameters['img_size'][0]
    return patch_size

def build_segdict_composite( img_metadata, boxes, segmentation_records, line_attributes, contour_tolerance=4.0, line_height_factor=1.0):
    """
    Construct the region + line segmentation dictionary.

    Args:
        img_metadata (dict): original image's metadata.
        boxes (list[tuple]): list of LTRB coordinate vectors, one for each region.
        segmentation_records (list[tuple[np.ndarray, list[tuple]]]): a list of N tuples (one per region) with
            - label map (np.ndarray)
            - a list of line attribute dicts (label, centroid pt, ..., area, polygon_coords)
        contour_tolerance (float): value for contour approximation (default: 4)

    Return:
        dict: a segmentation dictionary
    """
    segdict = { 'created': str(datetime.now()), 'creator': __file__, }
    segdict.update( img_metadata )
    segdict['line_height_factor']=line_height_factor
    segdict['regions']=[]

    region_id = 0
    for box, record in zip(boxes, segmentation_records):
        this_region_lines = []
        line_id = 0
        _, atts = record
        offset = np.array([box[1],box[0]])
        centroid_ys = [] 
        for att_dict in atts:
            label, polygon_coords, centroid,line_height, centerline, baseline = [ att_dict[k] for k in ('label','polygon_coords','centroid','line_height', 'centerline', 'baseline')]
            # adding box offsets
            polygon_coords += offset.astype(polygon_coords.dtype)
            centroid_ys.append( centroid[0].item() )
            baseline += offset.astype(baseline.dtype)
            centerline += offset.astype(centerline.dtype)
            dict_line_entry = {'id': f'l{line_id}', 'coords': polygon_coords[:,::-1].astype('int').tolist(), 'baseline': baseline[:,::-1].astype('int').tolist() }
            if 'height' in line_attributes:
                dict_line_entry['height']=int(line_height)
            if 'centerline' in line_attributes:
                dict_line_entry['centerline']=centerline[:,::-1].tolist() # yx to xy
            this_region_lines.append( dict_line_entry )
            line_id += 1
        line_spacings = np.array(centroid_ys[1:]) - np.array(centroid_ys[:-1])
        line_spacing_avg, line_spacing_min, line_spacing_max, line_spacing_std = [ int(v.item()) for v in (np.mean(line_spacings), np.min(line_spacings), np.max(line_spacings), np.std(line_spacings)) ] if len(centroid_ys)>1 else (-1,-1,-1,-1)
        segdict['regions'].append( { 
            'id': f'r{region_id}', 'type': 'text_region', 
            'coords': [[int(pt[0]),int(pt[1])] for pt in ([box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]])], 
            'line_spacing': {'avg': line_spacing_avg, 'min': line_spacing_min, 'max': line_spacing_max, 'std': line_spacing_std }, 
            'lines': this_region_lines } )
        region_id += 1

    return segdict


def pack_fsdb_inputs_outputs( args:dict, segmentation_suffix:str ) -> list[tuple]:
    """
    Compile image files and/or charter paths in the CLI arguments.
    No existence check on the dependency (segmentation path).

    Args:
        dict: the parsed arguments.
        segmentation_suffix (str): suffix of the expected segmentation file.
    Returns:
        list[tuple]: a list of triplets (<img file path>, <segmentation file path>, <output file path>)
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
        segfile_path = Path( re.sub(r'{}$'.format( args.img_suffix), segmentation_suffix, str(img_path) ))
        output_dir = img_path.parent if not args.output_dir else Path(args.output_dir)
        path_triplets.append( ( img_path, segfile_path, output_dir.joinpath( f'{img_stem}.{args.appname}.pred.{args.output_format}')))
    #return path_triplets
    return sorted( path_triplets, key=lambda x: str(x))


if __name__ == "__main__":

    args, _ = fargv.fargv( p )

    if not args.region_classes:
        logger.info("The 'region_classes' parameter must contain at least one valid region name (from the layout app).a)")
        sys.exit()

    if not Path( args.model_path ).exists():
        raise FileNotFoundError("Could not find model file", args.model_path)
    live_model = lsg.SegModel.load( args.model_path ) 

    if args.raw_polygons and args.line_height_factor != 1.0:
        logger.warning("'-raw_polygons' option set: ignoring the line height factor ({}).".format( args.line_height_factor))

    # Store aggregate computation time for every batch of <args.timer> images 
    timer_means = []
    start_time = time()
    timer_logs = sys.stdout
    if args.timer > 0 and args.timer_logs != 'stdout':
        try:
            timer_logs = open( args.timer_logs, 'w') 
            timer_logs.write("ImageIndex\tAvg/{}\tRunningAvg\n".format(args.timer))
            timer_logs.close()
        except IOError as e:
            logger.warning("Failed to open timer logs '{}'".format( timer_logs ))


    charter_iterator = pack_fsdb_inputs_outputs( args, args.layout_suffix )
    for img_idx, img_triplet in enumerate( charter_iterator ):
        img_path, layout_file_path, output_file_path = img_triplet
        logger.debug( "File path={}".format( img_triplet[0]))
        if not args.overwrite_existing and output_file_path.exists():
            continue
        try:
            with Image.open( img_path, 'r' ) as img:

                img_metadata = { 'image_filename': str(img_path.name), 'image_width': img.size[0], 'image_height': img.size[1] }
                binary_mask, segdict = None, {}

                if not layout_file_path.exists():
                    logger.warning("{}\tCould not find layout segmentation file {}. Skipping item.".format( img_path, layout_file_path.name ))
                    continue
                
                with open(layout_file_path, 'r') as regseg_if:
                    regseg = json.load( regseg_if )
                    # extract crops from layout analysis file
                    layout_data = seglib.layout_regseg_to_crops( img, regseg, args.region_classes )
                    if not layout_data:
                        #logger.warning("Could not find region with name in {} in the layout segmentation file {}. Skipping item.".format( args.region_classes, layout_file_path ))
                        continue
                    crops_pil, boxes, classes = seglib.layout_regseg_to_crops( img, regseg, args.region_classes, force_rgb=True )

                    binary_masks = []
                    for crop_idx, crop_whc in enumerate(crops_pil):
                        binary_mask = None
                        # Inference from fixed-size patches
                        patch_size = check_patch_size_against_model( live_model, args.patch_size )
                        binary_mask = lgm.binary_mask_from_fixed_patches( crop_whc, patch_size=patch_size, model=live_model, mask_threshold=args.mask_threshold, box_threshold=args.box_threshold, device='cpu' if args.device=='cpu' else 'cuda' )
                        if binary_mask is None:
                            logger.warning("{}\tNo line mask found in crop {}: skipping item.".format( img_path, crop_idx ))
                            continue
                        binary_masks.append( binary_mask )
                    try:
                        segmentation_records = [ lgm.get_morphology( msk, raw_polygons=args.raw_polygons, height_factor=args.line_height_factor ) for msk in binary_masks ]
                        segdict = build_segdict_composite( img_metadata, boxes, segmentation_records, args.line_attributes ) 
                    except (TypeError,ValueError) as e:
                        logger.warning("{}\tFailed to polygonize line masks ({}): abort segmentation.".format( img_path, e ))
                        continue

                ############ Output #################
                logger.debug(f"Serializing segmentation for img.shape={img.size}")

                if args.output_format == 'stdout':
                    print(json.dumps(segdict))
                if not output_file_path.exists() or args.overwrite_existing:
                    if args.output_format == 'json':
                        with open(output_file_path, 'w') as of:
                            #segdict['image_wh']=img.size
                            of.write(json.dumps( segdict, indent=4 ))
                    elif args.output_format == 'xml':
                        #segdict['image_wh']=img.size
                        seglib.xml_from_segmentation_dict( segdict, pagexml_filename=output_file_path )
                    logger.debug("Segmentation output saved in {}".format( output_file_path ))

                if args.timer > 0 and img_idx > 0 and img_idx % args.timer==0:
                    timer_means.append( (time()-start_time)/args.timer )
                    running_avg = np.mean(timer_means)
                    if timer_logs is sys.stdout:
                        logger.info( "Batch {}/{} (size={}): {:.4f}s/img\t Running avg: {:.4f}\tEst. time to completion: {}".format( 
                                    int(img_idx/args.timer), 
                                    int(np.ceil(len(charter_iterator)/args.timer)), 
                                    args.timer, 
                                    timer_means[-1], 
                                    running_avg, 
                                    duration_estimate(img_idx, len(charter_iterator), running_avg)))
                    else:
                        with open( timer_logs, 'a') as timer_of:
                            timer_of.write( "{}\t{:.4f}\t{:.4f}\n".format( img_idx, timer_means[-1], np.mean(timer_means)))
                    start_time = time()

        except Exception as e:
            logger.warning("{}".format( e ))
            continue
