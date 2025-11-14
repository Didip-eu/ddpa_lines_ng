#!/usr/bin/env python3
"""
A stand-alone script for PageXML -> JSON conversion.

The original function is in ddpa_lines_ng/libs/seglib.
"""

import sys
import json
import fargv
import re
from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Union, Any



src_root = Path(__file__).parents[1]
sys.path.append( str( src_root ))
from libs import seglib


p = {
    'file_paths': set([]),
    'output_format': ('json', 'stdout'),
    'get_text': [1, "Extract text content of the line, if it exists"],
    'overwrite_existing': [0, "Overwrite an existing file."],
    "comment": ['',"A text string to be added to the <Comments> elt."],
}


if __name__ == '__main__':

    args, _ = fargv.fargv( p )

    for xml_path in args.file_paths:

        xml_path = Path(xml_path)

        segdict = seglib.segmentation_dict_from_xml( xml_path, get_text=args.get_text )
        segdict_str = json.dumps( segdict, indent=2 )

        if args.output_format == 'stdout':
            print( segdict_str )
        else:
            json_path = xml_path.with_suffix('.json')
            if not args.overwrite_existing and json_path.exists():
                print("File {} exists: abort.".format( json_path ))
            else:
                with open(json_path, 'w') as json_outf:
                    json_outf.write( segdict_str )

