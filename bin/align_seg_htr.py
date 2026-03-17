#!/usr/bin/env python3
"""
Given line prediction file (JSON) and an inherited HTR GT (PageXML),
compute an alignment of predicted lines with the corresponding HTR strings.

Example:

```
$ align_seg_htr.py U-17_0728_r.lines.pred.json 
```

Output:

```
0 Wir, ✳ nachgenempten ✳ Clewi Merck von ✳ Schintznach, Hans Kilcher, ...
1 lich mitt disem brief, ✳ als den̄ ettwas zweyung zwischent den erwird...
2 uff hùtt, datū dis briefs, inder guͤtlikeit ✳ zuͦtz und uff uns komen ...
3 ten ✳ ir zuͦsprùch zuͦ den genempten ✳ unsern frowen ingegēwùrtikeit i...
4 nung, ✳ die selben unser frowen ✳ oder ir amptlùt hettent ein weg, d...
...

Rules and assumptions:

+ inherited (PageXML) segmentation must be valid (no partial or invalid polygons, f.i.)
+ polygons are matched by finding pairs whose baseline 
    - quadratic fits are closest
    - baseline length close (+/- 10%)
```

TODO:

+ Remove pairings that duplicate reference lines

"""


import json
import sys
from pathlib import Path
import numpy as np
from numpy.polynomial import Polynomial, polynomial
import matplotlib.pyplot as plt
import fargv

from libs import seglib
import warnings 
#warnings.simplefilter('ignore', np.exceptions.RankWarning)

import logging

logging_format="%(asctime)s - %(levelname)s: %(funcName)s - %(message)s"
logging_levels = {0: logging.ERROR, 1: logging.WARNING, 2: logging.INFO, 3: logging.DEBUG }
logging.basicConfig( level=logging.INFO, format=logging_format, force=True )
logger = logging.getLogger(__name__)

logging.getLogger('PIL').setLevel(logging.INFO)


p = {
        "segfile_path": ['', "A JSON line segmentation file (e.g <prefix>.lines.pred.json)."],
        "htr_path": ['', "A PageXML file with HTR strings (e.g. <prefix>.xml)."],
        "output_file": ['stdout', "Output file (eg. <prefix>.htr.aligned.json)."],
        "overwrite_existing": [0, "Do not overwrite existing output file"],
        "verbosity": [2, "Verbosity levels: 0 (quiet), 1 (WARNING), 2 (INFO-default), 3 (DEBUG)"],
        "matching_iou": [.15, "Tolerance for lengths of matching baselines."],
        'visual_check': [0, "Plot baseline estimate for visual checking."],
    }


def iou( box1: np.ndarray, box2: np.ndarray ):
	x1 = max(box1[0], box2[0])
	y1 = max(box1[1], box2[1])
	x2 = min(box1[2], box2[2])
	y2 = min(box1[3], box2[3])
	interArea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
	box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
	box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
	iou = interArea / float(box1Area + box2Area - interArea)
	return iou

def plot_polynoms( set1_l2n: np.ndarray, set2_l2n: np.ndarray, labels=('predicted', 'reference') ):
    """
    Args:
        set1_l2n (np.ndarray): (L,2,N) array of function values
        set2_l2n (np.ndarray): (L,2,N) array of function values
    """
    plt.close()
    fig, ax = plt.subplots()
    ax.plot( *(set1_l2n[0]), color='red', label=labels[0])
    ax.plot( *(set2_l2n[0]), color='green', label=labels[1])
    ax.yaxis.set_inverted(True)
    for l in set1_l2n[1:]:
        ax.plot( *l, color='red')
    for l in set2_l2n[1:]:
        ax.plot( *l, color='green')
    ax.legend()
    plt.show()


def polynoms_from_lines( lines, domain, window ):
    polynoms = []
    for l in lines:
        bl_arr = np.array(l['baseline'])
        domain = bl_arr[[0,-1],0]
        deg = 1 if len(bl_arr)<4 else 2
        polynoms.append( Polynomial.fit( *(bl_arr.T), deg=deg, domain=domain, window=window) )
    return polynoms

if __name__ == "__main__":

    args_orig = sys.argv.copy() # keep for logging
    args, _ = fargv.fargv( p )
    if args.verbosity != 2:
        logging.basicConfig( level=logging_levels[args.verbosity], format=logging_format, force=True )

    logger.debug( args.segfile_path )

    output_file_path = Path( args.segfile_path.replace('.lines.pred.json', '.htr.gt.json')) 

    if args.output_file != 'stdout' and not args.overwrite_existing and output_file_path.exists():
        logger.info(f"Existing {output_file_path}: skipping." )
        sys.exit()

    prediction_dict = json.load(open( args.segfile_path ))

    reference_file_path = Path(args.segfile_path.replace('.lines.pred.json', '.xml')) if not args.htr_path else Path( args.htr_path )
    if not reference_file_path.exists():
        raise FileNotFoundError()
    reference_dict = seglib.segmentation_dict_from_xml( reference_file_path, get_text=True )
    reference_dict = seglib.segdict_sink_lines( reference_dict)
    #print(reference_dict)

    # 1. Match regions
    pred_reg_to_ref_reg = []
    for r1 in prediction_dict['regions']:
        for r2 in reference_dict['regions']:
            box1 = np.array( r1['coords'][0]+r1['coords'][2] )
            box2 = np.array( r2['coords'][0]+r2['coords'][2] )
            if iou( box1, box2 ) > .85:
                pred_reg_to_ref_reg.append( r2 )
                break

    # 2. For each region: match lines

    for pred_reg_idx, ref_reg in enumerate(pred_reg_to_ref_reg):
        domain=prediction_dict['regions'][pred_reg_idx]['coords'][0][0], prediction_dict['regions'][pred_reg_idx]['coords'][2][0]
        window=prediction_dict['regions'][pred_reg_idx]['coords'][0][1], prediction_dict['regions'][pred_reg_idx]['coords'][2][1]

        predicted_lines = prediction_dict['regions'][pred_reg_idx]['lines'] 
        reference_lines = pred_reg_to_ref_reg[pred_reg_idx]['lines']


        predicted_polynomials = polynoms_from_lines( predicted_lines, domain, window )
        reference_polynomials = polynoms_from_lines( reference_lines, domain, window )

        matches = []

        predicted_polynom_points = np.stack([ p.linspace(5) for p in predicted_polynomials ])
        reference_polynom_points = np.stack([ p.linspace(5) for p in reference_polynomials ])

        if args.visual_check:
            plot_polynoms( predicted_polynom_points, reference_polynom_points )

        for p,ppn in enumerate(predicted_polynomials):
            for r,rpn in enumerate(reference_polynomials):
                length_p, length_r = [ l['baseline'][-1][0]-l['baseline'][0][0] for l in (predicted_lines[p], reference_lines[r]) ]
                score = ((predicted_polynom_points[p][1] - reference_polynom_points[r][1])**2).sum()
                matches.append((p, r, length_p, length_r, score ))

        # keeping best <predicted number> matches, minus the ill-fitting ones (insufficient overlap)
        matches = [ m for m in sorted( matches, key=lambda x: x[4])[:len(predicted_polynomials)] if abs((m[2] - m[3])/m[2]) <= .15]
        logger.debug(f"{len(matches)} matches.")
        match_hash = {}
        for m in sorted( matches, key=lambda x: x[0]):
            if m[0] not in match_hash:
                match_hash[m[0]] = m
        for k,m in match_hash.items():
            logger.debug(f"{m}")

        # check that no given reference line is assigned to 2 predicted lines
        assigned_reference_lines = set([ m[1] for m in match_hash.values() ])
        if len(assigned_reference_lines) != len( match_hash.keys()):
            logger.info("Assignment is not one to one! Removing sub-optimal pairs.")
            # remove pairs that duplicate a reference line
            continue
        # cases to handle:
        # - a predicted line has no counterpart in the original
        predicted_orphans = [ i for i, m in enumerate(matches) if m[0]==-1 ]
        if predicted_orphans:
            logger.warning(f"Predicted lines {predicted_orphans} have no match!")
        # - a reference line has no counterpart in the prediction
        difference = set( range( len( reference_lines ))) - assigned_reference_lines
        if difference:
            logger.warning(f"Reference lines {difference} have no counterpart in predictions.")
        
        for m in sorted(matches, key=lambda x: x[0]):
            if m[0]>=0:
                pred_lidx, ref_lidx, length_p , length_r, _= m
                logger.debug(pred_lidx, ref_lidx, ref_reg['lines'][ref_lidx]['text'][:100], length_p, length_r)
                prediction_dict['regions'][pred_reg_idx]['lines'][pred_lidx]['text']=ref_reg['lines'][ref_lidx]['text']

        cli_args = ' '.join(args_orig[1:])
        prediction_dict['metadata']['comment']=f"Created by command: {Path(args_orig[0]).name + cli_args} (input PageXML: {reference_file_path.name})."

        if args.output_file == 'stdout':
            print(json.dumps( prediction_dict ))
        else:
            with open( output_file_path, 'w') as of:
                of.write( json.dumps( prediction_dict ))


