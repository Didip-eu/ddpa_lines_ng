#!/usr/bin/env python3
"""
From a raw model pickle (as resulting from training), write a production
pickle, with built-in box and mask thresholding values."
"""

import torch
import hashlib
import sys
import shutil
from pathlib import Path


USAGE=f"{sys.argv[0]} <model_file> <threshold_dict_string>"

if len(sys.argv) < 3:
    print(USAGE)
    sys.exit()


orig_md5 = ''
with open(sys.argv[1], 'rb') as pickle_as_bytes:
    orig_md5 = hashlib.md5( pickle_as_bytes.read()).hexdigest()

orig_m = torch.load( sys.argv[1], weights_only=False)


orig_m.pop('epochs', None)
threshold_dict = eval(sys.argv[2])
invalid_keys = [ k for k in threshold_dict.keys() if k not in ('mask_threshold', 'box_threshold', 'comment')]
if invalid_keys:
    print(f"Incorrect threshold keys: {invalid_keys}: aborting.")
    sys.exit()

threshold_dict.update({'original_model_file': orig_md5})
orig_m['production_environment']=threshold_dict

torch.save( orig_m, 'mlmodel.tmp' )

with open('mlmodel.tmp', 'rb') as new_pickle_as_bytes:
    new_md5 = hashlib.md5( new_pickle_as_bytes.read()).hexdigest()
    confirm = input("Write new pickle? [Y|n]")
    if (not confirm) or confirm == 'Y':
        shutil.copyfile('mlmodel.tmp', f'{new_md5}.mlmodel')
        if Path(f'{new_md5}.mlmodel').exists():
            print(f"Packaged model: {new_md5}.mlmodel")

Path('mlmodel.tmp').unlink(missing_ok=True)





