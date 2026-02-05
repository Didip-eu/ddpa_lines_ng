#!/usr/bin/env python3
"""
JSON -> JSON conversion

Read a JSON segmentation file, with a choice of options:

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
import fargv
import re
from datetime import datetime
from pathlib import Path

src_root = Path(__file__).parents[1]
sys.path.append( str( src_root ))
from libs import seglib

p = {
    'file_path': ['', "Input file."],
    'polygon_key': 'coords',
    'line_height_factor': [1.0, "Factor to be applied to the original line strip height."],
    'output_file': ['', "Output file (default: standard output)."],
    'overwrite_existing': [0, "Overwrite an existing output file."],
    'drop_transcription': [0, "Extract line transcription, if it exists."],
    'promote_regions': [0, "For each region, create one separate file."],
    'delete_line_features': [set(), "Line items to be removed (used with caution!)"],
    "comment": ['',"A text string to be added to the <Comments> elt."],
}


if __name__ == '__main__':

    args, _ = fargv.fargv( p )

    if not args.file_path:
        sys.exit()

    json_path = Path( args.file_path )

    if args.promote_regions:
        from PIL import Image
        for reg_idx, region_tuple in enumerate( seglib.promote_regions_from_json_file( json_path )):
            region_img, region_segdict = region_tuple
            #region_img.show( region_img )
            # construct image and json file names
            output_json_path = re.sub( r'\.lines.*\.json$', f".r{reg_idx}"+r'\g<0>', args.file_path )
            output_img_path = json_path.parent.joinpath( region_segdict['image_filename'] )
            with open( output_json_path,'w') as of:
                of.write( json.dumps( region_segdict, indent=2))
                print(f"Compiled region file {output_json_path}")
            region_img.save( output_img_path )
            print(f"Saved region image {output_img_path}")
            sys.exit()

    output_path = None
    if args.output_file: 
        output_path = Path( args.output_file )
        if not args.overwrite_existing and output_path.exists():
            print("File {} exists: abort.".format(args.output_file))
            sys.exit()

    segdict = None

    with open( json_path, 'r') as json_if:
        segdict = json.load( json_if )

        # automatic 
        if 'lines' in segdict:
            segdict = seglib.segdict_sink_lines( segdict )

        line_dicts = seglib.line_dicts_from_segmentation_dict( segdict )

        # delete unwanted features
        for line in line_dicts:
            for key in args.delete_line_features:
                if key not in line:
                    continue
                del line[key]


        # expand polygons
        if args.line_height_factor != 1.0:
            line_polygons = seglib.line_polygons_from_segmentation_dict( segdict, polygon_key=args.polygon_key, factor=args.line_height_factor )
            for polyg, line in zip( line_polygons, line_dicts ):
                line[args.polygon_key]=polyg
        # remove transcriptions
        if args.drop_transcription:
            for line in line_dicts: 
                if 'text' in line:
                    del line['text']

        # insert metadata at the top
        regions = segdict['regions']
        del segdict['regions']
        segdict['metadata'].update( {'created': str(datetime.now()), 'creator': __file__ })

        if args.line_height_factor != 1.0:
            segdict['line_height_factor']=args.line_height_factor
        if args.comment:
            segdict['metadata']['comments']=args.comment
        segdict['regions']=regions




        # output
        if segdict is not None:
            if args.output_file:
                with open( output_path,'w') as of:
                    of.write( json.dumps( segdict, indent=2))
            else:
                print( json.dumps( segdict, indent=2 ))

