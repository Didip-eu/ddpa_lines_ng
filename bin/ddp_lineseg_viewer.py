#!/usr/bin/env python3

# nprenet@gmail.com
# 05.2025

"""
A simple line segmenter/viewer that predicts lines on images and displays the result.

A segmenter just for the eyes, or a viewer that can segment if needed: it stands by itself, with no regard for export functionalities.
For proper segmentation and recording of a region-based segmentation (crops), see `ddp_line_detect.py`, that is meant
to included into an HTR pipeline.

Examples:

    With image used as-is (no layout analysis), on-the-fly prediction and display:

    ```
    PYTHONPATH=. bin/ddp_lineseg_viewer.py -random 10 -model_path ./models/best_101_1024_bsz4.mlmodel -rescale 1 -img_paths ./dataset/*.jpg 
    ```

    Display an existing segmentation:

    ```
    PYTHONPATH=. bin/ddp_lineseg_viewer.py -random 10 -segfile_suffix lines.pred.json  -img_paths ./dataset/*.jpg
    ```

    High-quality segmentation of a COUS (Charter of Unusual Size), running inference separately on 3x1 patches:

    ```
    PYTHONPATH=. bin/ddp_lineseg_viewer.py -img_paths data/hard_cases/591e0762397178ee89e4c8b356be0da3.Wr_OldText.3.img.jpg -model_path ./models/best_101_1024_bsz4.mlmodel -patch_row_count 3
    ```

For proper segmentation and recording of a region-based segmentation (crops), see `ddp_line_detect.py`.'


"""

# stdlib
from pathlib import Path
import time
import sys
import random
import logging
import itertools
import math

# 3rd party
import matplotlib.pyplot as plt
from PIL import Image
import skimage as ski
import numpy as np
import torch

# DiDip
import fargv

# local
src_root = Path(__file__).parents[1]
sys.path.append( str( src_root ))
import ddp_lineseg as lsg
from libs import segviz, list_utils as lu




logging.basicConfig( level=logging.DEBUG, format="%(asctime)s - %(levelname)s: %(funcName)s - %(message)s", force=True )
logger = logging.getLogger(__name__)
# tone down unwanted logging
logging.getLogger('matplotlib.font_manager').disabled=True
logging.getLogger('PIL').setLevel(logging.INFO)


p = {
    'model_path': str(src_root.joinpath("best.mlmodel")),
    'mask_threshold': [0.25, "Threshold used for line masks--a tweak on the post-processing phase."],
    'rescale': [0, "If True, display segmentation on original image; otherwise (default), get the image size from the model used for inference (ex. 1024 x 1024)."],
    'img_paths': set(Path('dataset').glob('*.jpg')),
    'color_count': [0, "Number of colors for polygon overlay: -1 for single color, n > 1 for fixed number of colors, 0 for 1 color/line."],
    'limit': [0, "How many files to display."],
    'random': [0, "If non-null, randomly pick <random> paths out of the <img_paths> list."],
    'segfile_suffix': ['', "If a line segmentation suffix is provided (ex. 'lines.pred.json'), predicted lines are read from <img_path>.<suffix>."],
    'segfile': ['', "If a line segmentation file is provided, predicted lines are read from this file."],
    'patch_row_count': [ 0, "Process the image in <patch_row_count> rows."],
    'patch_col_count': [ 0, "Process the image in <patch_col_count> cols."],
    'patch_size': [0, "Process the image by <patch_size>*<patch_size> patches"],
}



def label_map_from_patches( img: Image.Image, row_count=2, col_count=1, overlap=50, model=None):
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
    logger.info("row_count={}, col_count={}".format(row_count, col_count))
    row_cuts_exact, col_cuts_exact  = [ list(int(f) for f in np.linspace(0, dim, d)) for dim, d in ((img.height, row_count+1), (img.width, col_count+1)) ]
    row_cuts, col_cuts = [[[ c+overlap, c-overlap] if c and c<cuts[-1] else c for c in cuts ] for cuts in ( row_cuts_exact, col_cuts_exact ) ]
    rows, cols = [ lu.group( lu.flatten( cut ), gs=2) for cut in (row_cuts, col_cuts) ]
    crops_yyxx=[ lu.flatten(lst) for lst in itertools.product( rows, cols ) ]
    logger.info(crops_yyxx)
    img_hwc = np.array( img )
    img_crops = [ torch.from_numpy(img_hwc[ crop[0]:crop[1], crop[2]:crop[3] ]).permute(2,0,1) for crop in crops_yyxx ]
    _, crop_preds, crop_sizes = lsg.predict( img_crops, live_model=model )
    page_mask = np.zeros((crops_yyxx[-1][1],crops_yyxx[-1][3]), dtype='bool')
    for i in range(len(crops_yyxx)):
        t,b,l,r = crops_yyxx[i]
        patch_mask = lsg.post_process( crop_preds[i], orig_size=crop_sizes[i], mask_threshold=.2 )
        if patch_mask is None:
            continue
        page_mask[t:b, l:r] += patch_mask[0]
    return page_mask[None,:]


def tile_img( img_hwc:np.ndarray, size, constraint=20 ):
    height, width = img_hwc.shape[:2]
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
        overlap = (row*size - height)//(row-1)
        y_pos = [ r*(size-overlap) if r < row-1 else height-size for r in range(row) ]

    return list(itertools.product(y_pos, x_pos ))

def label_map_from_fixed_patches( img: Image.Image, patch_size=1024, overlap=100, model=None):
    """
    Construct a single label map from predictions on patches of size <patch_size> x <patch_size>.

    Args:
        img (Image.Image): a PIL image.
        patch_size (int): size of the square patch.
        overlap (int): minimum overlap between patches (in pixels)

    Returns:
        np.ndarray: a (1,H,W) label map.
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
    tile_tls = tile_img( img_hwc, patch_size, constraint=overlap )
    img_crops = [ torch.from_numpy(img_hwc[y:y+patch_size,x:x+patch_size]).permute(2,0,1) for (y,x) in tile_tls ]
    logger.debug([ c.shape for c in img_crops ])
    
    _, crop_preds, _ = lsg.predict( img_crops, live_model=model )
    page_mask = np.zeros((img_hwc.shape[0],img_hwc.shape[1]), dtype='bool')
    for i,(y,x) in enumerate(tile_tls):
        patch_mask = lsg.post_process( crop_preds[i], mask_threshold=.2 )
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

    live_model = lsg.SegModel.load( args.model_path ) if (not args.segfile_suffix and not args.segfile) else None

    files = []
    if args.random:
        files = random.sample([ Path(p) for p in args.img_paths ], args.random)
    else:
        files = [ Path(p) for p in ( list(args.img_paths)[:args.limit] if args.limit else args.img_paths) ]

    for img_path in files:
        logger.info(img_path)

        if live_model:
            start = time.time()
            mp, atts, path = None, None, None
            start = time.time()
            if args.patch_row_count or args.patch_col_count:
                patch_row_count = args.patch_row_count if args.patch_row_count else 1
                patch_col_count = args.patch_col_count if args.patch_col_count else 1
                logger.debug("Patches: {}x{}".format(patch_row_count, patch_col_count))
                label_mask = label_map_from_patches( Image.open(img_path), patch_row_count, patch_col_count, model=live_model )
                logger.debug("Inference time: {:.5f}s".format( time.time()-start))
                logger.debug("label_mask.shape={}".format(label_mask.shape))
                segmentation_record = lsg.get_morphology( label_mask, centerlines=False)
                logger.debug("segmentation_record[0].shape={}".format(segmentation_record[0].shape))
                mp, atts, path = segviz.batch_visuals( [img_path], [segmentation_record], color_count=0 )[0]
            elif args.patch_size:
                logger.debug('Patch size: {} x {}'.format( args.patch_size, args.patch_size))
                label_mask = label_map_from_fixed_patches( Image.open(img_path), patch_size=args.patch_size, model=live_model )
                logger.debug("Inference time: {:.5f}s".format( time.time()-start))
                logger.debug("label_mask.shape={}".format(label_mask.shape))
                segmentation_record = lsg.get_morphology( label_mask, centerlines=False)
                mp, atts, path = segviz.batch_visuals( [img_path], [segmentation_record], color_count=0 )[0]
            else:
                imgs_t, preds, sizes = lsg.predict( [img_path], live_model=live_model)
                logger.debug("Inference time: {:.5f}s".format( time.time()-start))
                if args.rescale:
                    logger.debug("Rescale")
                    label_mask = lsg.post_process( preds[0], orig_size=sizes[0], mask_threshold=args.mask_threshold )
                    if label_mask is None:
                        logger.warning("No line mask found for {}: skipping.".format( img_path ))
                        continue
                    logger.debug("label_mask.shape={}".format(label_mask.shape))
                    segmentation_record = lsg.get_morphology( label_mask, centerlines=False)
                    mp, atts, path = segviz.batch_visuals( [img_path], [segmentation_record], color_count=0 )[0]
                else:
                    logger.debug("Square")
                    label_mask = lsg.post_process( preds[0], mask_threshold=args.mask_threshold )
                    if label_mask is None:
                        logger.warning("No line mask found for {}: skipping.".format( img_path ))
                        continue
                    segmentation_records= lsg.get_morphology( lsg.post_process( preds[0], mask_threshold=args.mask_threshold )) 
                    mp, atts, path = segviz.batch_visuals( [ {'img':imgs_t[0], 'id':str(img_path)} ], [segmentation_records], color_count=0 )[0]
            logger.debug("Rendering time: {:.5f}s".format( time.time()-start))

            plt.imshow( mp )
            plt.title( path )
            for att_dict in atts:
                label, centroid = att_dict['label'], att_dict['centroid']
                plt.text(*centroid[:0:-1], label, size=15)
            plt.show()
        else:
            if args.segfile:
                segviz.display_segmentation_and_img( img_path, segfile=args.segfile, regions=True )
            elif args.segfile_suffix:
                segviz.display_segmentation_and_img( img_path, segfile_suffix=args.segfile_suffix, regions=True )

