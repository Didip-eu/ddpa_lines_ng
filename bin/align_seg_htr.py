#!/usr/bin/env python3


import json
import sys
from pathlib import Path
import numpy as np
from numpy.polynomial import Polynomial, polynomial

from libs import seglib


prediction_dict = json.load(open(sys.argv[1]))

reference_file_path = Path(sys.argv[1].replace('.lines.pred.json', '.xml')) if len(sys.argv)<3 else Path(sys.argv[2])
if not reference_file_path.exists():
    raise FileNotFoundError()
reference_dict = seglib.segmentation_dict_from_xml( reference_file_path )
#print(reference_dict)

predicted_lines = [ l for r in prediction_dict['regions'] for l in r['lines']]
predicted_polynomials = [ Polynomial.fit( *(np.array(l['baseline']).T), deg=2 ) for l in predicted_lines ]
print(predicted_polynomials)


reference_lines = reference_dict['lines']
reference_polynomials = [ Polynomial.fit( *(np.array(l['baseline']).T), deg=2 ) for l in reference_lines ]
print(reference_polynomials)

matches = []
for p,ppn in enumerate(predicted_polynomials):
    for r,rpn in enumerate(reference_polynomials):
        matches.append(p,r,polynomial.polysub(ppn,rpn))
print(matches)

