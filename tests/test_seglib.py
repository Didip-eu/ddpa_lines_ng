import pytest
import json
import sys
import copy
from pathlib import Path
import numpy as np
from torch import Tensor

sys.path.append( str(Path(__file__).parents[1]))

@pytest.fixture(scope="session")
def data_path():
    return Path( __file__ ).parent.joinpath('data')


from libs import segformats as sgf, seglib


# 5 regions including one empty region, 7 lines
@pytest.fixture
def regular_dict():
    return {
      'metadata': {'created': '2026-05-06 14:02:38.619923',
      'creator': '/home/nicolas/graz/htr/vre/ddpa_htr/libs/seglib.py'},
     'type': 'baselines',
     'text_direction': 'horizontal-lr',
     'image_filename': '214_b088d_default.jpg',
     'image_width': 3812,
     'image_height': 5634,
     'regions': [{'id': 'eSc_textblock_15073fca',
       'coords': [[3177, 375], [3341, 375], [3341, 464], [3177, 464]],
       'lines': [{'id': 'eSc_line_64b6f04c',
         'baseline': [[3172, 450], [3341, 450]],
         'coords': [[3172, 450], [3182, 384], [3290, 356], [3332, 366], [3332, 469], [3276, 460], [3252, 483]],
         'x-height': 10,
         'text': '103'}]},
      {'id': 'eSc_textblock_2cdb1f5e',
       'coords': [[230, 520], [1462, 520], [1462, 4395], [230, 4395]],
       'lines': [{'id': 'eSc_line_92d04678',
         'baseline': [[282, 671], [389, 668], [472, 665], [1414, 638]],
         'coords': [[277, 666], [277, 507], [343, 511], [455, 600], [554, 591], [582, 568], [653, 596]],
         'x-height': 12,
         'text': 'Auoit non lemouicina.Etliruissiaus'},
        {'id': 'eSc_line_5b44302e',
         'baseline': [[286, 760], [1433, 727]],
         'coords': [[286, 760], [291, 708], [338, 694], [930, 694], [1109, 671], [1269, 685], [1363, 666]],
         'x-height': 12,
         'text': 'dela fonteine coroit par les iardins ⁊par les'},
        {'id': 'eSc_line_c430c007',
         'baseline': [[282, 990], [1400, 953]],
         'coords': [[282, 990], [286, 943], [314, 939], [681, 934], [836, 910], [1113, 920], [1386, 896]],
         'x-height': 11,
         'text': 'conduit ⁊mout en auoient grantioie.'}]},
      {'id': 'eSc_textblock_0f7e0cee',
       'coords': [[1542, 453], [2902, 453], [2902, 4395], [1542, 4395]],
       'lines': [{'id': 'eSc_line_39053db2',
         'baseline': [[1654, 652], [2754, 652]],
         'coords': [[1654, 652], [1663, 492], [1753, 492], [1804, 539], [1856, 530], [1931, 568], [2049, 549]],
         'x-height': 11,
         'text': 'Puet len souent abien uenir.Qui bien'},
        {'id': 'eSc_line_7df0805b',
         'baseline': [[1668, 962], [2749, 962]],
         'coords': [[1668, 962], [1678, 910], [1974, 896], [2688, 896], [2740, 910], [2749, 962], [2730, 981]],
         'x-height': 10,
         'text': 'ture.por ce se doit len au bien auoier :⁊'}]},
      {'id': 'eSc_textblock_1524f9d9',
       'coords': [[984, 162], [1714, 162], [1714, 347], [984, 347]],
       'lines': [{'id': 'eSc_line_536efca2',
         'baseline': [[1006, 306], [1694, 302]],
         'coords': [[1005, 305], [1010, 220], [1048, 197], [1118, 239], [1240, 239], [1330, 173], [1396, 206]],
         'x-height': 13,
         'text': 'MARTIN'}]},
      {'id': 'eSc_textblock_b16cd7c7',
       'coords': [[1534, 2854], [1964, 2854], [1964, 3163], [1534, 3163]],
       'lines': []}]}

# 3 upper-level regions, 2 nested regions, 7 lines
@pytest.fixture
def nested_regions_dict( regular_dict ):
    nested_dict = copy.deepcopy( regular_dict )
    inner_regions = nested_dict['regions'][2:4]
    nested_dict['regions'][1]['regions']=inner_regions
    nested_dict['regions'].pop(3)
    nested_dict['regions'].pop(2)
    return nested_dict

def test_line_binary_mask_stack_from_segmentation_dict( regular_dict ):
    binary_mask_stack = seglib.line_binary_mask_stack_from_segmentation_dict( regular_dict )
    assert type(binary_mask_stack) is Tensor
    assert binary_mask_stack.shape == (7, 5634, 3812)

def test_line_images_from_img_json_files( data_path ):
    img_path = data_path.joinpath('217_d9c7f_default.jpg')
    json_path = data_path.joinpath('217_d9c7f_default.json')
    seglib.line_images_from_img_json_files( img_path, json_path )
    assert True

def test_line_images_from_img_page_xml_files( data_path ):
    img_path = data_path.joinpath('217_d9c7f_default.jpg')
    page_xml_path = data_path.joinpath('217_d9c7f_default.page.xml')
    seglib.line_images_from_img_page_xml_files( img_path, page_xml_path )
    assert True

def test_line_polygons_from_segmentation_dict( regular_dict ):
    polygons = seglib.line_polygons_from_segmentation_dict( regular_dict )
    assert len(polygons)==7

def test_line_polygons_from_segmentation_dict_scaled( regular_dict ):
    polygons = seglib.line_polygons_from_segmentation_dict( regular_dict )
    scaled_polygons = seglib.line_polygons_from_segmentation_dict( regular_dict, factor=1.25 )
    assert len(scaled_polygons)==7
    assert scaled_polygons != polygons

def test_promote_regions_bad_file( data_path ):
    """Fail on improper file (lines not within their intended regions)"""
    json_path = str(data_path.joinpath('217_d9c7f_default.json'))
    with pytest.raises(AssertionError):
        seglib.promote_regions_from_json_file( json_path )

def test_promote_regions( data_path ):
    json_path = str(data_path.joinpath('4550c41616f9ef3a1a95a94155e67fe9.json'))
    assert seglib.promote_regions_from_json_file( json_path )

