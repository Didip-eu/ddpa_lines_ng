#!/usr/bin/env python3
"""
Search local FSDB tree for charters that match an charter md5 id, or an image md5 id.
"""

import sys
import argparse
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

from libs import fsdb

if __name__ == '__main__':

    parser =argparse.ArgumentParser()

    fsdb_root_arg = parser.add_argument("--fsdb_root", type=str, default=str(Path.home().joinpath('tmp/data/fsdb')))
    parser.add_argument("--img_id", type=str)
    parser.add_argument("--charter_id", type=str)

    args = parser.parse_args()

    if not Path(args.fsdb_root).exists():
        raise argparse.ArgumentError(fsdb_root_arg, f"'{args.fsdb_root}' is not an existing path. Abort.")
        sys.exit(-1)

    fsdb_inst = fsdb.Fsdb( args.fsdb_root )

    print( json.dumps( list([ ch.to_json() for ch in fsdb_inst.get_charter( charter_img_id='99d808a952c0870ceb7a237456d6b7a3') ])))
    
