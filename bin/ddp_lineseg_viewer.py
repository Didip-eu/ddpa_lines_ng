#!/usr/bin/env python3

# nprenet@gmail.com
# 05.2025

"""
A simple line segmenter/viewer that predicts lines on images and displays the result.

Examples:

    With image used as-is (no layout analysis), on-the-fly prediction and display:

    ```
    PYTHONPATH=. bin/ddp_lineseg_viewer.py -random 10 -model_path ./models/best_101_1024_bsz4.mlmodel -rescale 1 -img_paths ./dataset/*.jpg 
    ```

    Display an existing segmentation:

    ```
    PYTHONPATH=. bin/ddp_lineseg_viewer.py -random 10 -segfile_suffix lines.pred.json  -img_paths ./dataset/*.jpg
    ```

For proper segmentation and recording of a region-based segmentation (crops), see `ddp_line_detect.py`.'

TODO:
    - display an existing segmentation

"""

# stdlib
from pathlib import Path
import time
import sys
import random
import logging

# 3rd party
import matplotlib.pyplot as plt

# DiDip
import fargv

# local
src_root = Path(__file__).parents[1]
sys.path.append( str( src_root ))
import ddp_lineseg as lsg
from libs import segviz




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
    "centerlines": [0, "If True, compute centerlines (default is False)."],
    'limit': [0, "How many files to display."],
    'random': [0, "If non-null, randomly pick <random> paths out of the <img_paths> list."],
    'segfile_suffix': ['', "If a line segmentation suffix is provided (ex. 'lines.pred.json'), predicted lines are read from <img_path>.<suffix>."],
    'segfile': ['', "If a line segmentation file is provided, predicted lines are read from this file."],
}


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
            imgs_t, preds, sizes = lsg.predict( [img_path], live_model=live_model)

            logger.debug("Inference time: {:.5f}s".format( time.time()-start))

            maps = []
            start = time.time()
            if args.rescale:
                maps=[ lsg.post_process( p, orig_size=sz, mask_threshold=args.mask_threshold, centerlines=args.centerlines ) for (p,sz) in zip(preds,sizes) ]
                mp, atts, path = segviz.batch_visuals( [img_path], maps, color_count=0 )[0]
            else:
                maps=[ lsg.post_process( p, mask_threshold=args.mask_threshold, centerlines=args.centerlines ) for p in preds ]
                mp, atts, path = segviz.batch_visuals( [ {'img':imgs_t[0], 'id':str(img_path)} ], maps, color_count=0 )[0]
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

