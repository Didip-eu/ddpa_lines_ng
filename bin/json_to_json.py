#!/usr/bin/env python3
"""
JSON -> JSON conversion

Read a JSON segmentation file, with a choice of options:

+ repair a faulty file
+ expand polygon heights wr/ original x-height
+ remove transcription data
+ add a comment
+ for each region, create 1+ new file(s)

Legacy format (with a top-level 'lines' array) is silently converted to
the nested structure

    { 'regions': [
        { 'coords': [ ... ],
          'lines': [{ ... }, ... ] }, ...
      ]
    }

"""

import sys
import json
import re
from datetime import datetime
from pathlib import Path
import logging

import fargv
from fargv import FargvPositional, FargvChoice

from segtformats import segtformats as sgf

src_root = Path(__file__).parents[1]
sys.path.append( str( src_root ))
from libs import seglib

p = {
    'file_paths': FargvPositional(default=[], description="Input file (JSON)."),
    'out': FargvChoice(['','auto'], description="Set to 'auto' for output to filename <input stem>.<output_suffix>; leave empty for standard output."),
    'line_height_factor': (1.0, "Factor to be applied to the original line strip height."),
    'input_suffix': ('.lines.pred.json', "Input file suffix."),
    'output_suffix': '.json',
    'overwrite_existing': (False, "Overwrite an existing output file."),
    'drop_transcription': (False, "Drop line transcription, if they exist."),
    'promote_regions': (False, "For each region, create one separate file (pre-pend 'r<reg_nbr>' to input suffix."),
    'repair': (False, "Repair a faulty dictionary: re-assign lines to their proper regions; expand regions to include every pixel of the line polygon."),
    'delete_line_features': ('', "Line attributes to be removed, as a colon-separated string of keys. E.g. 'x-height:centerline'."),
    "comment": ('',"A text string to be added to the <Comments> elt."),
    'verbosity': FargvChoice(['2','0','1','3'], description="Verbosity levels: 0 (quiet), 1 (WARNING), 2 (INFO-default), 3 (DEBUG)"),
}

logging_format="%(asctime)s - %(levelname)s: %(filename)s: %(funcName)s - %(message)s"
logger = logging.getLogger(__name__)


if __name__ == '__main__':

    args, _ = fargv.parse( p )

    logging.basicConfig(
            level={0: logging.ERROR, 1: logging.WARNING, 2: logging.INFO, 3: logging.DEBUG }[int(args.verbosity)],
            format=logging_format,
            force=True )

    for file_path in args.file_paths:
        json_path = Path( file_path )

        segdict = None
        with open( json_path, 'r') as json_if:
            segdict = json.load( json_if )

            if args.repair:
                logger.info("Repairing file")
                segdict = sgf.json_doctor( segdict )

            # region-as-a-file extraction
            if args.promote_regions:
                from PIL import Image
                for reg_idx, region_tuple in enumerate( seglib.promote_regions_from_segdict( segdict, json_path.parent )):
                    region_img, region_segdict = region_tuple
                    #region_img.show( region_img )
                    # construct image and json file names
                    output_json_path = args.file_path.replace( args.input_suffix, f".r{reg_idx}" + args.input_suffix )
                    #re.sub( r'\.lines.*\.json$', f".r{reg_idx}"+r'\g<0>', args.file_path )
                    if not args.overwrite_existing and Path(output_json_path).exists():
                        continue
                    output_img_path = json_path.parent.joinpath( region_segdict['image_filename'] )
                    with open( output_json_path,'w') as of:
                        of.write( json.dumps( region_segdict, indent=2))
                        logger.info(f"Compiled region file {output_json_path}")
                    region_img.save( output_img_path )
                    logger.info(f"Saved region image {output_img_path}")
                continue

            line_dicts = sgf.line_dicts_from_segmentation_dict( segdict, by_reference=True )

            # delete unwanted features
            if args.delete_line_features:
                for line in line_dicts:
                    for key in args.delete_line_features.split(':'):
                        if key not in line:
                            continue
                        del line[key]

            # expand polygons
            if args.line_height_factor != 1.0:
                line_polygons = seglib.line_polygons_from_segmentation_dict( segdict, factor=args.line_height_factor )
                for polyg, line in zip( line_polygons, line_dicts ):
                    line['coords']=polyg
            # remove transcriptions
            if args.drop_transcription:
                for line in line_dicts:
                    if 'text' in line:
                        del line['text']

            # insert metadata at the top
            regions = segdict['regions']
            del segdict['regions']
            segdict['metadata'].update( {'created': str(datetime.now().isoformat('T','seconds')), 'creator': __file__ })

            if args.line_height_factor != 1.0:
                segdict['line_height_factor']=args.line_height_factor
            if args.comment:
                segdict['metadata']['comments']=args.comment
            segdict['regions']=regions

            segdict_str = json.dumps( segdict, indent=2 )
            if not args.out:
                print( segdict_str )
                continue

            out_path = ''
            if args.out == 'auto':
                if not re.search( r'{}$'.format(args.input_suffix), json_path.name):
                    logger.warning(f"Input file path '{json_path.name}' does not match input suffix '{args.input_suffix}': output aborted.")
                    continue
                out_path = re.sub(r'{}$'.format( args.input_suffix ), args.output_suffix, json_path )
            logger.debug(f"Output file = {out_path}")

            if not args.overwrite_existing and Path(out_path).exists():
                logger.info("File {} exists: skipping.".format( out_path ))
                continue

            with open(out_path, 'w') as json_outf:
                logger.info(f"W: {json_path.name} → {Path(out_path).name}")
                json_outf.write( segdict_str )

