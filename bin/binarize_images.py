#!/usr/bin/env python3

from PIL import Image
from pathlib import Path
import numpy as np
from libs import seglib

for path in Path('dataset').glob('*.jpg'):
    mask_path = Path('dataset/binary/{}.bin.npy'.format( path.name.replace('.img.jpg', '') ))
    if mask_path.exists():
        continue
    print(mask_path)
    img = Image.open( path, 'r' )
    mask = seglib.get_binary_mask( img )
    np.save(mask_path, mask)

