#!/usr/bin/env python3
"""
JSON -> JSON conversion

Read a JSON segmentation file, with a choice of options:

+ expand polygon heights wr/ original x-height
+ remove transcription data
+ add a comment

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
from pathlib import Path

src_root = Path(__file__).parents[1]
sys.path.append( str( src_root ))
from libs import seglib

p = {
    'file_path': ['', "Input file."],
    'polygon_key': 'coords',
    'line_height_factor': [1.0, "Factor to be applied to the original line strip height."],
    'output_file': ['stdout', "Output file"],
    'overwrite_existing': [0, "Overwrite an existing output file."],
    'with_transcription': [1, "Extract line transcription, if it exists"],
    "comment": ['',"A text string to be added to the <Comments> elt."],
}


if __name__ == '__main__':

    args, _ = fargv.fargv( p )

    if not args.file_path:
        sys.exit()

    json_path = Path( args.file_path )
    output_path = None
    if args.output_file != 'stdout':
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

        # expand polygons
        if args.line_height_factor != 1.0:
            line_polygons = seglib.line_polygons_from_segmentation_dict( segdict, polygon_key=args.polygon_key, factor=args.line_height_factor )
            for polyg, line in zip( line_polygons, line_dicts ):
                line[args.polygon_key]=polyg
        # remove transcriptions
        for line in line_dicts: 
            if not args.with_transcription and 'text' in line:
                del line['text']

        # insert metadata at the top
        regions = segdict['regions']
        del segdict['regions']
        if args.line_height_factor != 1.0:
            segdict['line_height_factor']=args.line_height_factor
        if args.comment:
            segdict['comment']=args.comment
        segdict['regions']=regions

    # output
    if segdict is not None:
        if args.output_file != 'stdout':
            with open( output_path,'w') as of:
                of.write( json.dumps( segdict, indent=2))
        else:
            print( json.dumps( segdict, indent=2 ))

