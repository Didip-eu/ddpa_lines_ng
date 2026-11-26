"""
Regression testing on line detection:

+ boundary cases:
  - img with no layout
  - layout with no OldText or region
  - region with no lines
+ advanced features (polygon scaling or dilation) fail gently
+ return value or exceptions on improper input

The tests use pre-computed layout files. All images are to be found in the FSDB tree.
"""
