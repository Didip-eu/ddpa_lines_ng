#!/usr/bin/env python3
"""
Given line prediction file (JSON) and an inherited HTR GT (PageXML),
compute an alignment of predicted lines with the corresponding HTR strings.

Output:

```
0 Wir, ✳ nachgenempten ✳ Clewi Merck von ✳ Schintznach, Hans Kilcher, ...
1 lich mitt disem brief, ✳ als den̄ ettwas zweyung zwischent den erwird...
2 uff hùtt, datū dis briefs, inder guͤtlikeit ✳ zuͦtz und uff uns komen ...
3 ten ✳ ir zuͦsprùch zuͦ den genempten ✳ unsern frowen ingegēwùrtikeit i...
4 nung, ✳ die selben unser frowen ✳ oder ir amptlùt hettent ein weg, d...
...
```

TODO:

+ library functions
+ usable output (array)
+ warning for region with no match (in either input file)

"""


import json
import sys
from pathlib import Path
import numpy as np
from numpy.polynomial import Polynomial, polynomial

from libs import seglib


USAGE=f"USAGE: {sys.argv[0]} <segfile>.lines.pred.json [ <segfile>.xml ]"

if len(sys.argv)<2 or sys.argv[1]=='-h' or sys.argv[2]=='-h':
    print(USAGE)
    sys.exit()

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


prediction_dict = json.load(open(sys.argv[1]))

reference_file_path = Path(sys.argv[1].replace('.lines.pred.json', '.xml')) if len(sys.argv)<3 else Path(sys.argv[2])
if not reference_file_path.exists():
    raise FileNotFoundError()
reference_dict = seglib.segmentation_dict_from_xml( reference_file_path, get_text=True )
reference_dict = seglib.segdict_sink_lines( reference_dict)
print(reference_dict)
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
    #print("domain={}, window={}".format( domain, window ))
    predicted_lines = prediction_dict['regions'][pred_reg_idx]['lines'] 
    predicted_polynomials = [ Polynomial.fit( *(np.array(l['baseline']).T), deg=2, domain=domain, window=window ) for l in predicted_lines ]
    #print(predicted_polynomials)


    reference_lines = pred_reg_to_ref_reg[pred_reg_idx]['lines']
    reference_polynomials = [ Polynomial.fit( *(np.array(l['baseline']).T), deg=2, domain=domain, window=window ) for l in reference_lines ]
    #print(reference_polynomials)

    matches = []
    for p,ppn in enumerate(predicted_polynomials):
        for r,rpn in enumerate(reference_polynomials):
            _, pred_y = ppn.linspace(50) 
            _, ref_y = rpn.linspace(50) 
            matches.append((p,r,((pred_y-ref_y)**2).sum()))
    best_matches=sorted(matches,key=lambda x: x[2])[:len(predicted_polynomials)]
    for m in sorted(best_matches, key=lambda x: x[0] ):
        pred_lidx, ref_lidx, _ = m
        print(pred_lidx, ref_reg['lines'][ref_lidx]['text'])

