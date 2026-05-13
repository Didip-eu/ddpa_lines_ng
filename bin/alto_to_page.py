#!/usr/bin/env python3
"""
ALTO → Page conversion tool with embedded XSL stylesheet.
"""

import sys
import re
from pathlib import Path

from libs import segformats

USAGE=f"USAGE: {sys.argv[0]} <alto file>.xml [<PageXML output file]"

if len(sys.argv) < 2 or re.match(r'--?h', sys.argv[1]):
    print(USAGE)
    sys.exit()

source_file = sys.argv[1]

if len(sys.argv)>2 and Path(sys.argv[2]).exists():
    segformats.alto_to_page_xml( source_file, pagexml_filename=sys.argv[2] )
else:
    print( segformats.alto_to_page_xml( source_file, as_string=True ) )
