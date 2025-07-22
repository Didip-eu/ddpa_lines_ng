#!/usr/bin/env python3
"""
Compute and print average line height, given a segmentation file.
"""

import skimage as ski
import numpy as np
from pathlib import Path
import sys

sys.path.append( str(Path(__file__).parents[1] ))
from libs import seglib



def plot_line_heights( json_segfile ):
    mask = seglib.line_binary_mask_from_json_file( json_segfile ).numpy()
    label_map = ski.measure.label( mask, connectivity=2 )
    heights = np.array( [ reg.area/reg.axis_major_length for reg in ski.measure.regionprops( label_map ) if reg.axis_major_length > 0 ])
    return np.mean(heights)



if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print('USAGE: {} <json_segmentation_file>'.format( sys.argv[0] ))
        sys.exit()

    print('{:.1f}'.format( plot_line_heights( sys.argv[1] )), end='')

