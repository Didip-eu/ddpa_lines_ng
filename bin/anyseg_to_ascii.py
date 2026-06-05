#!/usr/bin/env python3
"""
ASCII rendition of a segmentation data file.
"""

import sys
import re
import tty
import termios
from pathlib import Path

import fargv
from fargv import FargvChoice, FargvInt, FargvFloat, FargvPositional, FargvVariadic

from libs import segformats

p = {
    'file_paths': FargvVariadic([], description="Input file (JSON, Page, Alto)."),
    "lines": FargvChoice(['1','2','0'], description="0=lines omitted, 1=lines within region limits, 2=lines within canvas limits"),
    "scale": (1.0, "Factor to be applied to the default scale."),
}   


if __name__ == '__main__':

    # line-oriented (default) → char-oriented input
    setting = termios.tcgetattr(sys.stdin.fileno())
    tty.setcbreak(sys.stdin)

    args, _ = fargv.parse( p )
    if not args.file_paths:
        print("Input file name expected! Abort.")
        sys.exit()
    for file_path in args.file_paths:
        print( segformats.any_to_ascii( file_path, lines=int(args.lines), scale_hw=args.scale ))
        print("\n[Next file: 'q' or 'n']")
        while True:
            q=sys.stdin.read(1)
            if q in ['q', 'n', 'Q', 'N']:
                break
    termios.tcsetattr( sys.stdin, termios.TCSADRAIN, setting )
        
