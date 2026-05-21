#!/usr/bin/env python3

# nprenet@gmail.com
# 05.2026

"""
Line detection script, not intended for production use, but for evaluation purpose: 

+ inference is run on single image-as-a-region (no layout metadata expected)
+ use arbitrary image paths (no FSDB assumed)
+ guaranteed to return a valid segmentation dictionary (even with no lines)

Output formats: 

+ PageXML: core polygon and baseline only.
+ JSON: custom format, including features that are not in the PageXML spec: centerline, line height.

Example call:
    
    PYTHONPATH=. python3 ./bin/ddp_line_detect_test.py --img_paths dataset/test/*.img.jpg --output_dir dataset/test/predictions --device gpu

Notes:

+ GPU can be an issue on very large images---use PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
"""
# stdlib
import sys
from pathlib import Path
import json
import re
import sys
from datetime import datetime
import logging
from time import time
import traceback

# 3rd party
from PIL import Image, UnidentifiedImageError
import numpy as np

# Didip
import fargv
from fargv import FargvChoice, FargvInt, FargvFloat, FargvPositional, FargvTuple

# local

src_root = Path(__file__).parents[1]
sys.path.append( str( src_root ))
from libs.train_utils import duration_estimate
from libs import segmodel as sgm, line_geometry as lgm, seglib
from libs import segformats as sgf

logging_format="%(asctime)s - %(levelname)s: %(funcName)s - %(message)s"
logging_levels = {0: logging.ERROR, 1: logging.WARNING, 2: logging.INFO, 3: logging.DEBUG }
logging.basicConfig( level=logging.INFO, format=logging_format, force=True )
logger = logging.getLogger(__name__)

logging.getLogger('PIL').setLevel(logging.INFO)


p = {
        "appname": "lines",
        "model_path": str(src_root.joinpath("best.mlmodel")),
        "img_paths": FargvPositional(default=[]),
        "img_suffix": (r".img.*p*g", "Image file suffix."),
        "output_suffix": '',
        "line_attributes": (["centerline", "x-height"], "Non-standard line properties to be included in the dictionary."),
        "output_format": FargvChoice(["json", "xml", "docufcn", "stdout", "quiet"], description="Segmentation output: json=<JSON file>, xml=<PageXML file>, stdout=JSON on standard output, quiet=nothing (for testing and timing)"),
        "output_dir": ('', "Output directory; if not provided, defaults to the image path's parent."),
        'mask_threshold': (.6, "In the post-processing phase, threshold to use for line soft masks."),
        'box_threshold': (0.75, "Threshold used for line bounding boxes."),
        'apply_model_thresholds': (True, "If true, any threshold passed as parameter overrides model's built-in thresholds."),
        'patch_size': (1024, "Process the image by <patch_size>*<patch_size> patches"),
        'raw_polygons': (False, "Serialize polygons as resulting from the NN (default); otherwise, construct the abstract polygons from centerlines."),
        'device': FargvChoice(['cpu','gpu','cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'], description="Computing device -- 'cuda' or 'gpu' defaults to 'cuda:0'."),
        'line_height_factor': (1.0, "Factor (within (0,1]) to be applied to the polygon height: allows for extracting polygons that extend above and below the core line-unused if 'raw_polygons' set"),
        'overwrite_existing': (True, "Write over existing output file (default)."),
        'timer': (0, "Aggregate performance metrics. A strictly positive integer <n> computes the mean time for every batch of <n> images."),
        'timer_logs': ('stdout', "Filename for timer logs."),
        'verbosity': (2,"Verbosity levels: 0 (quiet), 1 (WARNING), 2 (INFO-default), 3 (DEBUG)"),
        'validate': (True, "Validate output against JSON schema;"),
}


def check_patch_size_against_model( live_model: dict, patch_size ):
    if 'train_style' in live_model.hyper_parameters:
        if live_model.hyper_parameters['train_style'] != 'patch':
           logger.warning('The model being loaded was _not_ trained on fixed-size patches: expect suboptimal results.')
        elif live_model.hyper_parameters['img_size'][0] != args.patch_size:
           logger.warning('The model being loaded is trained on {}x{} patches, but the script uses a {} patch size argument: overriding patch_size value with model-stored size.'.format( *live_model.hyper_parameters['img_size'], args.patch_size))
           return live_model.hyper_parameters['img_size'][0]
    return patch_size

def build_segdict( img_metadata, segmentation_record=None, line_attributes=['x-height', 'centerline'], contour_tolerance=4.0, line_height_factor=1.0):
    """
    Construct the region + line segmentation dictionary.

    Args:
        img_metadata (dict): original image's metadata.
        segmentation_records (list[tuple[np.ndarray, list[tuple]]]): a list of N tuples (one per region) with
            - label map (np.ndarray)
            - a list of line attribute dicts (label, centroid pt, ..., area, polygon_coords)
        contour_tolerance (float): value for contour approximation (default: 4)

    Return:
        dict: a segmentation dictionaryi, with image-as-a-region.
    """
    segdict = { 'metadata': {'created': str(datetime.now()), 'creator': __file__, }}
    segdict.update( img_metadata )
    segdict['line_height_factor']=line_height_factor
    
    segdict['regions']=[ {
        'coords': [[0,0],[img_metadata['image_width'],0],[img_metadata['image_width'], img_metadata['image_height']],[0,img_metadata['image_height']]] } ]

    if not segmentation_record:
        return segdict

    region_id = 0
    this_region_lines = []
    line_id = 0
    _, atts = segmentation_record
    centroid_ys = [] 
    for att_dict in atts:
        label, polygon_coords, centroid,line_height, centerline, baseline = [ att_dict[k] for k in ('label','polygon_coords','centroid','line_height', 'centerline', 'baseline')]
        centroid_ys.append( centroid[0].item() )
        dict_line_entry = {'id': f'l{line_id}', 'coords': polygon_coords[:,::-1].astype('int').tolist(), 'baseline': baseline[:,::-1].astype('int').tolist() }
        if 'x-height' in line_attributes:
            dict_line_entry['x-height']=int(line_height)
        if 'centerline' in line_attributes:
            dict_line_entry['centerline']=centerline[:,::-1].tolist() # yx to xy
        this_region_lines.append( dict_line_entry )
        line_id += 1
    line_spacings = np.array(centroid_ys[1:]) - np.array(centroid_ys[:-1])
    line_spacing_avg, line_spacing_min, line_spacing_max, line_spacing_std = [ int(v.item()) for v in (np.mean(line_spacings), np.min(line_spacings), np.max(line_spacings), np.std(line_spacings)) ] if len(centroid_ys)>1 else (-1,-1,-1,-1)
    segdict['regions'][0] = { 
        'id': f'r{region_id}', 'type': 'text_region', 
        'coords': [[0,0],[img_metadata['image_width'],0],[img_metadata['image_width'], img_metadata['image_height']],[0,img_metadata['image_height']]],
        'line_spacing': {'avg': line_spacing_avg, 'min': line_spacing_min, 'max': line_spacing_max, 'std': line_spacing_std }, 
        'lines': this_region_lines } 

    return segdict


def pack_inputs_outputs( args:dict ) -> list[tuple]:
    """
    Compile image files in the CLI arguments.
    No existence check on the dependency (layout segmentation path).

    Args:
        dict: the parsed arguments.
    Returns:
        list[tuple]: a list of pairs (<img file path>, <output file path>)
    """
    all_img_paths = set([ Path(p) for p in args.img_paths ])

    path_pairs = []
    if args.output_format == 'docufcn' and not args.output_suffix:
        logger.warning("No output suffix provided for chosen Doc-UFCN output format: using '.json'")
    for img_path in all_img_paths:
        img_stem = re.sub(r'{}$'.format( args.img_suffix), '', img_path.name )
        output_dir = img_path.parent if not args.output_dir else Path(args.output_dir)
        output_file_name = f'{img_stem}.stdout'
        if args.output_suffix:
            output_file_name = f'{img_stem}{args.output_suffix}'
        elif args.output_format=='xml' or args.output_format=='json':
            output_file_name = f'{img_stem}.{args.appname}.pred.{args.output_format}'
        elif args.output_format=='docufcn':
            output_file_name = f'{img_stem}.json'
        path_pairs.append( ( img_path, output_dir.joinpath( output_file_name )))
    return sorted( path_pairs, key=lambda x: str(x))


if __name__ == "__main__":

    args, _ = fargv.parse( p )

    if args.verbosity != 2:
        logging.basicConfig( level=logging_levels[args.verbosity], format=logging_format, force=True )

    if not Path( args.model_path ).exists():
        raise FileNotFoundError(args.model_path)

    thresholds = {'mask_threshold': args.mask_threshold, 'box_threshold': args.box_threshold }
    if args.apply_model_thresholds:
        thresholds = lgm.thresholds_from_model( args.model_path, thresholds )
    logger.debug(f"Thresholds = {thresholds}")

    live_model = sgm.SegModel.load( args.model_path ) 

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

    computing_device='cpu'
    if args.device == 'cuda' or args.device == 'gpu':
        computing_device='cuda:0'
    else:
        computing_device = args.device

    charter_iterator = pack_inputs_outputs( args )
    for img_idx, img_pair in enumerate( charter_iterator ):
        img_path, output_file_path = img_pair
        logger.info(f"image_path={img_path}")
        logger.debug( "output path={}".format(output_file_path))
        if not args.overwrite_existing and output_file_path.exists():
            logger.debug("File {} exists: skipped.".format( output_file_path ))
            continue
        try:
            with Image.open( img_path, 'r' ) as img:

                img_metadata = { 'image_filename': str(img_path.name), 'image_width': img.size[0], 'image_height': img.size[1] }
                binary_mask, segdict = None, {}

                # Inference from fixed-size patches
                patch_size = check_patch_size_against_model( live_model, args.patch_size )
                binary_mask = lgm.binary_mask_from_fixed_patches( img, patch_size=patch_size, model=live_model, mask_threshold=thresholds['mask_threshold'], box_threshold=thresholds['box_threshold'], device=computing_device)
                logger.debug(f"binary_mask={binary_mask}")
                if binary_mask is None:
                    logger.warning("{}\tNo line mask found in crop {}: skipping item.".format( img_path, crop_idx ))
                    # dict with no lines
                    segdict = build_segdict( img_metadata )
                try:
                    # Post-processing: pixel maps → lines & polygons
                    segmentation_record = lgm.get_morphology( binary_mask, raw_polygons=args.raw_polygons, height_factor=args.line_height_factor ) 
                    segdict = build_segdict( img_metadata, segmentation_record, args.line_attributes, line_height_factor=args.line_height_factor ) 
                    if args.validate and not sgf.json_validate( segdict ):
                        logger.error('Validation failed. Skipping item.')
                        continue
                except (TypeError, ValueError) as e:
                    logger.warning("{}\tFailed to polygonize line masks ({}): abort segmentation.".format( img_path, e ))
                    continue

                ############ Output #################

                sentinel_path = output_file_path.with_suffix('.stl') # protect against partial writes
                if args.output_format == 'stdout':
                    print(json.dumps(segdict))
                elif args.overwrite_existing or not output_file_path.exists() or sentinel_path.exists():
                    open( sentinel_path, 'w' )
                    if args.output_format == 'json' or args.output_format== 'docufcn':
                        if args.output_format == 'docufcn':
                            segdict = seglib.didip_json_to_docufcn_label_json( segdict )
                        with open(output_file_path, 'w') as of:
                            of.write(json.dumps( segdict, indent=4 ))
                    elif args.output_format == 'xml':
                        sgf.page_xml_from_segmentation_dict( segdict, pagexml_filename=output_file_path )
                    sentinel_path.unlink()
                    if args.output_format != 'quiet':
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
            logger.warning("{}: {}".format( img_path, e ))
            logger.warning(traceback.format_exc())
            continue
