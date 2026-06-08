#!/usr/bin/env python3
"""
ASCII rendition of segmentation data files.

Eg.::

    ┼┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┌────────────────┐┄┄┄┄┄┄┼
    ┆                  │r:2a93943e      │      ┆
    ┆     ┌────────────└──l:8┌────────────────┐┆
    ┆     │r:ef1657c4       ││r:ed8f3022      │┆
    ┆     │  l:1e355b0a...  ││  l:8590f3bf... │┆
    ┆     │  l:1e1c59e1.... ││  l:5c1a7591... │┆
    ┆     │  l:aa302e79...  ││  l:50a89193... │┆
    ┆     │  l:0bcc2987...  ││  l:ab831781... │┆
    ┆     │  l:f45a1fa8...  ││  l:b808a65a... │┆
    ┆     │  l:1d001e4b...  ││  l:640bb0ff....│┆
    ┆     │  l:e30c7a81..   │┌────┐0a959ab... │┆
    ┆     │  l:7bb2bb1f.... ││r:8fa3e8012b... │┆
    ┆     │  l:85832788...  ││  l:│7383a84... │┆
    ┆     ┌──────┐d5625...  │└────┘bd94110..  │┆
    ┆     │r:f77b369130..   ││  l:2b0c22bb... │┆
    ┆     │  l:a6│ecf48...  ││  l:1dac6035... │┆
    ┆     │  l:6f│ff3a2...  ││  l:acb1c73e... │┆
    ┆     └──────┘60b9c...  ││  l:5393c17b... │┆
    ┆     │  l:299e800c...  ││  l:4e171c9c... │┆
    ┆     │  l:508f7f75...  ││  l:d1ffc59c... │┆
    ┆     │  l:496470a0...  ││  l:5c56470d... │┆
    ┆     │  l:0ba0e73a...  ││  l:8a93cee6    │┆
    ┆     │  l:30b7e6ef...  ││  l:6d2db756    │┆
    ┆     │  l:e66fbbbb...  ││  l:b48006e0... │┆
    ┆     │  ...            ││  ...           │┆
    ┆     └─────────────────┘└────────────────┘┆
    ┆                                          ┆
    ┼┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┼


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
    "text": (False, "Display text, if present."),
}   

bold_start, bold_end = '\033[1m', '\033[0m'
help_msg = f"""
Key bindings: 

    {bold_start}n{bold_end}: {bold_start}n{bold_end}ext file
    {bold_start}p{bold_end}: {bold_start}p{bold_end}revious file
    {bold_start}l{bold_end}: circle through {bold_start}l{bold_end}ine display modes (1=region, 2=canvas, 0=none)
    {bold_start}r{bold_end}: {bold_start}r{bold_end}epair segmentation before rendering
    {bold_start}q{bold_end}: {bold_start}q{bold_end}uit application (or exit this help screen).
    {bold_start}t{bold_end}: display {bold_start}t{bold_end}ext.
    {bold_start}h{bold_end} or {bold_start}?{bold_end}: this {bold_start}h{bold_end}elp
"""

if __name__ == '__main__':


    args, _ = fargv.parse( p )
    if not args.file_paths:
        print("Input file name expected! Abort.")
        sys.exit()
    file_count=0
    lines = int(args.lines)
    repair = args.repair
    text = args.text
    help_screen = False

    try:
        # from line-oriented term (default) → char-oriented input
        setting = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin)

        while True:
            print('\x1b[2J')
            if help_screen:
                print(help_msg)
                q=sys.stdin.read(1)
                help_screen = False
                continue
            seg_rendition = segformats.any_to_ascii( args.file_paths[file_count], lines=lines, scale_hw=args.scale, repair=repair, text=text )
            seg_rendition_width = len(seg_rendition.split('\n')[-1])
            pagination=f"{file_count+1}/{len(args.file_paths)}" + (' [repaired]' if repair else '')
            footer_content=f"File {pagination}: {Path( args.file_paths[file_count] ).name}"
            footer = [' '] * seg_rendition_width
            footer[seg_rendition_width-( len(footer_content) + 4 ):seg_rendition_width-4]=footer_content
            print( seg_rendition )
            print(''.join(footer) )

            # Key bindings
            q=sys.stdin.read(1)
            if q == 'q':
                break
            elif q == 'n':
                file_count= (file_count+ 1) % len(args.file_paths)
                lines =  int(args.lines)
                repair = args.repair
            elif q == 'p':
                file_count= (file_count- 1) % len(args.file_paths)
                lines = int(args.lines)
                repair = args.repair
            elif q == 'l':
                lines = (lines + 1) % 3
            elif q == 'r':
                repair = not repair
            elif q == 't':
                text = not text
            elif q in ['h', '?']:
                help_screen = True
    finally:
        # no matter what, restore default terminal settings
        termios.tcsetattr( sys.stdin, termios.TCSADRAIN, setting )
        
