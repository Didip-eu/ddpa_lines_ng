#!/usr/bin/env python3
"""
PageXML -> JSON conversion.
"""

import sys
import json
import re
from pathlib import Path
from typing import Union, Any

import fargv
from fargv import FargvChoice, FargvPositional
from jsonschema import validate

src_root = Path(__file__).parents[1]
sys.path.append( str( src_root ))
from libs import seglib, segformats as sgf


p = {
    'file_paths': FargvPositional(default=[]),
    'output_format': FargvChoice(['json', 'stdout'], description="Output format"),
    'input_suffix': '.xml',
    'get_text': (True, "Extract text content of the line, if it exists"),
    'overwrite_existing': (False, "Overwrite an existing file."),
    "comment": ('',"A text string to be added to the <Comments> elt."),
    "verbose": False,
    "repair": (False, "Repair a faulty dictionary (reassign lines to regions, fix region bounding boxes"),
    "validate": (False, "Validate against a JSON schema."),
}


if __name__ == '__main__':

    args, _ = fargv.parse( p )

    for xml_path in args.file_paths:

        if args.verbose:
            print(xml_path)

        segdict = sgf.segmentation_dict_from_page_xml( xml_path, get_text=args.get_text )

        if args.repair:
            segdict = sgf.json_doctor( segdict, verbose=args.verbose )

        # Raise an exception if invalid
        if args.validate:
            sgf.json_validate( segdict )

        segdict_str = json.dumps( segdict, indent=2 )
        if args.output_format == 'stdout':
            print( segdict_str )
        else:
            json_path = Path(xml_path.replace(args.input_suffix, '.json'))
            if not args.overwrite_existing and json_path.exists():
                print("File {} exists: abort.".format( json_path ))
            elif not re.search( r'{}$'.format(args.input_suffix), Path(xml_path).name):
                print(f"Input file path '{xml_path.name}' does not match input suffix '{args.input_suffix}': output aborted.")
            else:
                with open(json_path, 'w') as json_outf:
                    json_outf.write( segdict_str )

