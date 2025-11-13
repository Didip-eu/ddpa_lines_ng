#!/usr/bin/env python3
"""
Script for JSON -> PageXML conversion.

"""

import sys
import json
import fargv
from pathlib import Path

src_root = Path(__file__).parents[1]
sys.path.append( str( src_root ))
from libs import seglib



p = {
    'file_paths': set([]),
    'polygon_key': 'coords',
    'output_format': ('xml', 'stdout'),
    'with_transcription': [1, "Extract line transcription, if it exists"],
    'line_height_factor': [1.0, "Factor to be applied to the original line strip height."],
    'comment': ['',"A text string to be added to the <Comments> elt."],
}


if __name__ == '__main__':

    args, _ = fargv.fargv( p )

    for json_path in args.file_paths:
        xml_path = json_path.replace('.lines.gt.json', '.xml')
        json_path = Path( json_path )

        with open( json_path, 'r') as json_if:
            segdict = json.load( json_if )
            if args.line_height_factor != 1.0:
                line_polygons = seglib.line_polygons_from_segmentation_dict( segdict, polygon_key=args.polygon_key, factor=args.line_height_factor )
                line_dicts = seglib.line_dicts_from_segmentation_dict( segdict )
                for polyg, line in zip( line_polygons, line_dicts ):
                    line[args.polygon_key]=polyg
                segdict['line_height_factor']=args.line_height_factor
            if args.comment:
                segdict['comment']=args.comment

            if args.output_format == 'stdout':
                seglib.xml_from_segmentation_dict( segdict, '', polygon_key=args.polygon_key, with_text=args.with_transcription )
            else:
                seglib.xml_from_segmentation_dict( segdict, xml_path, polygon_key=args.polygon_key, with_text=args.with_transcription )

