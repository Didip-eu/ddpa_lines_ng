#!/usr/bin/env python3
"""
ASCII rendition of a segmentation data file.
"""

import sys
import re
from pathlib import Path

import fargv
from fargv import FargvChoice, FargvInt, FargvFloat, FargvPositional, FargvVariadic

from libs import segformats

p = {
    'file_path': FargvVariadic([], description="Input file (JSON, Page, Alto)."),
    "lines": FargvChoice(['1','2','0'], description="0=lines omitted, 1=lines within region limits, 2=lines within canvas limits"),
    "scale": (1.0, "Factor to be applied to the default scale."),
}   


if __name__ == '__main__':

    args, _ = fargv.parse( p )
    if not args.file_path:
        print("Input file name expected! Abort.")
        sys.exit()
    print( segformats.any_to_ascii( args.file_path[0], lines=int(args.lines), scale_hw=args.scale ))
