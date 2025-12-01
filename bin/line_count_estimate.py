#!/usr/bin/env python3

# nprenet@gmail.com
# 11.2025

"""
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
from libs import segmodel as sgm

logging_format="%(asctime)s - %(levelname)s: %(funcName)s - %(message)s"
logging_levels = {0: logging.ERROR, 1: logging.WARNING, 2: logging.INFO, 3: logging.DEBUG }
logging.basicConfig( level=logging.INFO, format=logging_format, force=True )
logger = logging.getLogger(__name__)

logging.getLogger('PIL').setLevel(logging.INFO)


p = {
        "appname": "line_count",
        "model_path": str(src_root.joinpath("best.mlmodel")),
        "img_paths": set([]),
        "charter_dirs": set([]),
        "region_classes": [set(["Wr:OldText"]), "Names of the layout-app regions on which lines are to be detected. Eg. '[Wr:OldText']. If empty (default), detection is run on the entire page."],
        "img_suffix": [r".img.*p*g", "Image file suffix."],
        "layout_suffix": [".layout.pred.json", "Regions are given by segmentation file that is <img name stem><suffix>."],
        "sample_size": [300, "Sample image with  <sample_size> x <sample_size> square patches."],
        "sample_width": [0, "If strictly positive, sample image with  <sample_column> x <img_height> strips."],
        'verbosity': [2,"Verbosity levels: 0 (quiet), 1 (WARNING), 2 (INFO-default), 3 (DEBUG)"],
}


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
        path_triplets.append( ( img_path, segfile_path, Path( f'{img_stem}.{args.appname}.tsv')))
    #return path_triplets
    return sorted( path_triplets, key=lambda x: str(x))


if __name__ == "__main__":

    args, _ = fargv.fargv( p )

    if args.verbosity != 2:
        logging.basicConfig( level=logging_levels[args.verbosity], format=logging_format, force=True )

    if not args.region_classes:
        logger.info("The 'region_classes' parameter must contain at least one valid region name (from the layout app).a)")
        sys.exit()

    # Store aggregate computation time for every batch of <args.timer> images 
    charter_iterator = pack_fsdb_inputs_outputs( args, args.layout_suffix )

    print("Img\tGTCnt\tEst\tPredCnt\tDisp.\tl\tt\tr\tb")
    for img_idx, img_triplet in enumerate( charter_iterator ):
        img_path, layout_file_path, _ = img_triplet
        try:
            with Image.open( img_path, 'r' ) as img:

                ok = Path( str(img_path).replace('.img.jpg', '.seg_ok')).exists()
                gt_path = Path( str(img_path).replace('.img.jpg', '.lines.gt.json'))
                pregt_path = Path( str(img_path).replace('.img.jpg', '.lines.pregt.json'))
                pred_path = Path( str(img_path).replace('.img.jpg', '.lines.pred.json'))

                img_metadata = { 'image_filename': str(img_path.name), 'image_width': img.size[0], 'image_height': img.size[1] }

                if not layout_file_path.exists():
                    #logger.warning("{}\tCould not find layout segmentation file {}. Skipping item.".format( img_path, layout_file_path.name ))
                    continue
                line_estimates = [] 
                with open(layout_file_path, 'r') as regseg_if:
                    regseg = json.load( regseg_if )
                    # extract crops from layout analysis file
                    layout_data = seglib.layout_regseg_to_crops( img, regseg, args.region_classes )
                    if not layout_data:
                        #logger.warning("Could not find region with name in {} in the layout segmentation file {}. Skipping item.".format( args.region_classes, layout_file_path ))
                        continue
                    crops_pil, boxes, classes = seglib.layout_regseg_to_crops( img, regseg, args.region_classes, force_rgb=True )


                    #for crop_idx, crop_whc in enumerate(crops_pil):
                    biggest_reg_index = np.argmax([crop.size[0]*crop.size[1] for crop in crops_pil ])
                    crop_whc = crops_pil[ biggest_reg_index ]

                    gt_line_count, pred_line_count=-1,-1
                    if gt_path.exists() or pregt_path.exists():
                        path = gt_path if gt_path.exists() else pregt_path
                        seg_dict = json.load( open(path))
                        #print([ reg['coords'] for reg in seg_dict['regions']])
                        max_reg_idx = np.argmax( [ (reg['coords'][1][0]-reg['coords'][0][0])*(reg['coords'][2][1]-reg['coords'][1][1]) for reg in seg_dict['regions']])
                        gt_line_count = len(seg_dict['regions'][max_reg_idx]['lines'])
                    if pred_path.exists():
                        seg_dict = json.load( open(pred_path))
                        #print([ reg['coords'] for reg in seg_dict['regions']])
                        max_reg_idx = np.argmax( [ (reg['coords'][1][0]-reg['coords'][0][0])*(reg['coords'][2][1]-reg['coords'][1][1]) for reg in seg_dict['regions']])
                        pred_line_count = len(seg_dict['regions'][max_reg_idx]['lines'])

                    sample_size = args.sample_width if args.sample_width > 0 else (args.sample_size, args.sample_size)

                    count_hat, variance = lgm.line_count_estimate_ng( crop_whc, sample_size=sample_size, repeat=15 )
                    line_estimates.append((str(img_path), gt_line_count, count_hat, pred_line_count, variance, *boxes[ biggest_reg_index ]))

                ############ Output #################

                output_str = '\n'.join( [ f"{path}\t{gt_count}\t{count}\t{pred_count}\t{var:.3f}\t{int(l)}\t{int(t)}\t{int(r)}\t{int(b)}" for (path, gt_count, count, pred_count, var, l, t, r, b) in line_estimates ] )
                print( output_str )


        except Exception as e:
            logger.warning("{}".format( e ))
            continue
