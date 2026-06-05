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
from fargv import FargvChoice, FargvVariadic

from libs import segformats

p = {
    'file_paths': FargvVariadic([], description="Input file (JSON, Page, Alto)."),
    "lines": FargvChoice(['1','2','0'], description="0=lines omitted, 1=lines within region limits, 2=lines within canvas limits."),
    "scale": (1.0, "Factor to be applied to the default scale."),
    "repair": (False, "Try repairing a faulty segmentation before rendition (file is not modified)."),
}   

bold_start, bold_end = '\033[1m', '\033[0m'
help_msg = f"""
Key bindings: 

    {bold_start}n{bold_end}: {bold_start}n{bold_end}ext file
    {bold_start}p{bold_end}: {bold_start}p{bold_end}revious file
    {bold_start}l{bold_end}: circle through {bold_start}l{bold_end}ine display modes (1=region, 2=canvas, 0=none)
    {bold_start}r{bold_end}: {bold_start}r{bold_end}epair segmentation before rendering
    {bold_start}q{bold_end}: {bold_start}q{bold_end}uit application (or exit this help screen).
    {bold_start}h{bold_end} or {bold_start}?{bold_end}: this {bold_start}h{bold_end}elp
"""

if __name__ == '__main__':

    # line-oriented (default) → char-oriented input
    setting = termios.tcgetattr(sys.stdin.fileno())
    tty.setcbreak(sys.stdin)

    args, _ = fargv.parse( p )
    if not args.file_paths:
        print("Input file name expected! Abort.")
        sys.exit()
    i=0
    lines = int(args.lines)
    repair = args.repair
    help_screen = False

    try:
        while True:
            print('\x1b[2J')
            if help_screen:
                print(help_msg)
                q=sys.stdin.read(1)
                help_screen = False
                continue
            seg_rendition = segformats.any_to_ascii( args.file_paths[i], lines=lines, scale_hw=args.scale, repair=repair )
            seg_rendition_width = len(seg_rendition.split('\n')[-1])
            pagination=f"{i+1}/{len(args.file_paths)}" + (' [repaired]' if repair else '')
            footer_content=f"File {pagination}: {Path( args.file_paths[i] ).name}"
            footer = [' '] * seg_rendition_width
            footer[seg_rendition_width-( len(footer_content) + 4 ):seg_rendition_width-4]=footer_content
            print( seg_rendition )
            print(''.join(footer) )
            q=sys.stdin.read(1)
            if q == 'q':
                break
            elif q == 'n':
                i = (i + 1) % len(args.file_paths)
                lines =  int(args.lines)
                repair = args.repair
            elif q == 'p':
                i = (i - 1) % len(args.file_paths)
                lines = int(args.lines)
                repair = args.repair
            elif q == 'l':
                lines = (lines + 1) % 3
            elif q == 'r':
                repair = not repair
            elif q in ['h', '?']:
                help_screen = True
    finally:
        termios.tcsetattr( sys.stdin, termios.TCSADRAIN, setting )
        
