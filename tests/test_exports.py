import pytest
import json
import sys
from pathlib import Path

sys.path.append( str(Path(__file__).parents[1]))

from libs import seglib


top_lines_with_valid_regions={"image_filename": "020d26df5d8fcc91591d4919c0c1d03f.Wr_OldText.2.img.jpg", "image_width": 1405, "image_height": 983, 
 "lines": [
    {"id": 0, 
        "centerline": [[36, 67], [91, 66], [120, 69], [288, 69], [541, 63], [683, 57], [979, 59]],
        "baseline": [[36, 59], [91, 58], [120, 61], [288, 61], [541, 55], [683, 49]],
        "coords": [[1330, 99], [1279, 94], [1279, 94], [1279, 94], [1268, 93], [1267, 93]],
        "extBoundary": [[1329, 108], [1279, 103], [1278, 103], [1278, 103], [1267, 102]],
        "x-height": 19, 
        "region": 0},
    {"id": "l1", 
        "centerline": [[38, 109], [327, 111], [703, 103], [799, 103], [987, 104], [1086, 111]],
        "baseline": [[38, 101], [327, 103], [703, 95], [799, 95], [987, 96], [1086, 103]],
        "coords": [[1324, 141], [1282, 141], [1280, 141], [1277, 140], [1268, 134], [1265, 133]],
        "x-height": 19, 
        "region": "r1"}], 
  "regions": [{'id': 0, 'coords': [[38, 109], [327, 111], [703, 103], [799, 103]]}, {'id': 'r1', 'coords': [[38, 109], [327, 111], [703, 103], [799, 103]]}]
  }


top_lines_with_no_regions={"image_filename": "020d26df5d8fcc91591d4919c0c1d03f.Wr_OldText.2.img.jpg", "image_width": 1405, "image_height": 983,
 "lines": [
    {"id": 0, 
        "centerline": [[36, 67], [91, 66], [120, 69], [288, 69], [541, 63], [683, 57], [979, 59]],
        "baseline": [[36, 59], [91, 58], [120, 61], [288, 61], [541, 55], [683, 49]],
        "coords": [[1330, 99], [1279, 94], [1279, 94], [1279, 94], [1268, 93], [1267, 93]],
        "extBoundary": [[1329, 108], [1279, 103], [1278, 103], [1278, 103], [1267, 102]],
        "x-height": 19, },
    {"id": "l1", 
        "centerline": [[38, 109], [327, 111], [703, 103], [799, 103], [987, 104], [1086, 111]],
        "baseline": [[38, 101], [327, 103], [703, 95], [799, 95], [987, 96], [1086, 103]],
        "coords": [[1324, 141], [1282, 141], [1280, 141], [1277, 140], [1268, 134], [1265, 133]],
        "x-height": 19, }]
  }


top_regions={"image_filename": "020d26df5d8fcc91591d4919c0c1d03f.Wr_OldText.2.img.jpg", "image_width": 1405, "image_height": 983,
 "regions": [
     {"id": 0,
      "lines": [
           {"id": 0, 
            "centerline": [[36, 67], [91, 66], [120, 69], [288, 69], [541, 63], [683, 57], [979, 59]],
            "baseline": [[36, 59], [91, 58], [120, 61], [288, 61], [541, 55], [683, 49]],
            "coords": [[1330, 99], [1279, 94], [1279, 94], [1279, 94], [1268, 93], [1267, 93]],
            "x-height": 19, } ],
      "coords": [[38, 109], [327, 111], [703, 103], [799, 103]]},
     {"id": "r1",
      "lines": [
            {"id": "l1", 
             "centerline": [[38, 109], [327, 111], [703, 103], [799, 103], [987, 104], [1086, 111]],
             "baseline": [[38, 101], [327, 103], [703, 95], [799, 95], [987, 96], [1086, 103]],
             "coords": [[1324, 141], [1282, 141], [1280, 141], [1277, 140], [1268, 134], [1265, 133]],
             "x-height": 19, }],
      'coords': [[38, 109], [327, 111], [703, 103], [799, 103]]}]}


def test_xml_from_segmentation_dict():

    seglib.xml_from_segmentation_dict( top_lines_with_valid_regions )
    print('')
    seglib.xml_from_segmentation_dict( top_lines_with_no_regions )
    print('')
    seglib.xml_from_segmentation_dict( top_regions )

    
