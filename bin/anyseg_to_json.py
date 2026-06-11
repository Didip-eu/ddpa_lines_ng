#!/usr/bin/env python3
#
# Convert segmentation meta-data:
#  [ Alto, Page ] → JSON
#
#

import sys
from pathlib import Path
import json
import re

import fargv
from fargv import FargvChoice, FargvInt, FargvFloat, FargvPositional, FargvTuple

from libs import seglib, segformats as sgf

p = {
        "segfile_paths": FargvPositional(default=[], description="A JSON line segmentation file (e.g <prefix>.lines.pred.json)."),
        "overwrite_existing": (False, "Do not overwrite existing output file"),
        "validate": False,
        "output_format": FargvChoice(['json', 'stdout'], description="Output format"),
        "verbose": True,
}
args, _ = fargv.parse( p )


for segfile_path_str in args.segfile_paths:

    status_string=f"{segfile_path_str}: "

    segdict = {}
    seg_format = sgf.get_format( segfile_path_str )
    if seg_format==sgf.SegFormat.ALTO:
        segdict = sgf.alto_to_segmentation_dict( segfile_path_str )
    elif seg_format==sgf.SegFormat.PAGE:
        segdict = sgf.segmentation_dict_from_page_xml( segfile_path_str )
    elif seg_format==sgf.SegFormat.JSON:
        continue
    else:
        print("Unknown segmentation format: skipping item.", file=sys.stderr)
        continue

    if not segdict:
        if args.verbose:
            print( status_string + '____' )
        continue
    status_string += 'C'

    # repair, if needed
    segdict = sgf.json_doctor( segdict )
    status_string += 'R'

    if args.validate:
        if not sgf.json_validate( segdict ):
            if args.verbose:
                print( status_string + '__' )
            continue
        status_string += 'V'

    segdict_str = json.dumps( segdict, indent=2 )
    if args.output_format == 'stdout':
        print( segdict_str )
    else:
        json_path = re.sub(r'\.[^.]+$', r'.json', segfile_path_str )
        if not args.overwrite_existing and Path(json_path).exists():
            if args.verbose:
                print( status_string + "_ (file exists)".format( json_path ))
            continue
        with open(json_path, 'w') as json_outf:
            json_outf.write( segdict_str )
            if args.verbose:
                print( status_string + f"W → {json_path}")

