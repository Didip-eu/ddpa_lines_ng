import pytest
import json
import sys
import copy
from pathlib import Path

sys.path.append( str(Path(__file__).parents[1]))

@pytest.fixture(scope="session")
def data_path():
    return Path( __file__ ).parent.joinpath('data')


from libs import segformats as sgf

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
         'text': '103'}]},
      {'id': 'eSc_textblock_2cdb1f5e',
       'coords': [[230, 520], [1462, 520], [1462, 4395], [230, 4395]],
       'lines': [{'id': 'eSc_line_92d04678',
         'baseline': [[282, 671], [389, 668], [472, 665], [1414, 638]],
         'coords': [[277, 666], [277, 507], [343, 511], [455, 600], [554, 591], [582, 568], [653, 596]],
         'text': 'Auoit non lemouicina.Etliruissiaus'},
        {'id': 'eSc_line_5b44302e',
         'baseline': [[286, 760], [1433, 727]],
         'coords': [[286, 760], [291, 708], [338, 694], [930, 694], [1109, 671], [1269, 685], [1363, 666]],
         'text': 'dela fonteine coroit par les iardins ⁊par les'},
        {'id': 'eSc_line_c430c007',
         'baseline': [[282, 990], [1400, 953]],
         'coords': [[282, 990], [286, 943], [314, 939], [681, 934], [836, 910], [1113, 920], [1386, 896]],
         'text': 'conduit ⁊mout en auoient grantioie.'}]},
      {'id': 'eSc_textblock_0f7e0cee',
       'coords': [[1542, 453], [2902, 453], [2902, 4395], [1542, 4395]],
       'lines': [{'id': 'eSc_line_39053db2',
         'baseline': [[1654, 652], [2754, 652]],
         'coords': [[1654, 652], [1663, 492], [1753, 492], [1804, 539], [1856, 530], [1931, 568], [2049, 549]],
         'text': 'Puet len souent abien uenir.Qui bien'},
        {'id': 'eSc_line_7df0805b',
         'baseline': [[1668, 962], [2749, 962]],
         'coords': [[1668, 962], [1678, 910], [1974, 896], [2688, 896], [2740, 910], [2749, 962], [2730, 981]],
         'text': 'ture.por ce se doit len au bien auoier :⁊'}]},
      {'id': 'eSc_textblock_1524f9d9',
       'coords': [[984, 162], [1714, 162], [1714, 347], [984, 347]],
       'lines': [{'id': 'eSc_line_536efca2',
         'baseline': [[1006, 306], [1694, 302]],
         'coords': [[1005, 305], [1010, 220], [1048, 197], [1118, 239], [1240, 239], [1330, 173], [1396, 206]],
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


def test_get_format( data_path ):
    assert sgf.get_format( data_path.joinpath('217_d9c7f_default.alto.xml')) == sgf.SegFormat.ALTO
    assert sgf.get_format( data_path.joinpath('btv1b84473026_f25.chocomufin.xml')) == sgf.SegFormat.ALTO
    assert sgf.get_format( data_path.joinpath('217_d9c7f_default.page.xml')) == sgf.SegFormat.PAGE
    assert sgf.get_format( data_path.joinpath('217_d9c7f_default.json')) == sgf.SegFormat.JSON

def test_line_extraction_for_line_count_regular( regular_dict ):
    line_dicts = sgf.line_dicts_from_segmentation_dict( regular_dict )
    assert len(line_dicts)==7

def test_line_extraction_for_line_count_nested( nested_regions_dict ):
    line_dicts = sgf.line_dicts_from_segmentation_dict( nested_regions_dict )
    assert len(line_dicts)==7

def test_line_extraction_for_line_parent( regular_dict ):
    line_dicts = sgf.line_dicts_from_segmentation_dict( regular_dict )
    assert line_dicts[0]['parents']==['eSc_textblock_15073fca']
    assert all([ line_dicts[i]['parents']==['eSc_textblock_2cdb1f5e'] for i in range(1,4) ])

def test_line_extraction_for_line_parent_nested( nested_regions_dict ):
    line_dicts = sgf.line_dicts_from_segmentation_dict( nested_regions_dict )
    assert line_dicts[0]['parents']==['eSc_textblock_15073fca']
    assert all([ line_dicts[i]['parents']==['eSc_textblock_2cdb1f5e'] for i in range(1,4) ])
    assert all([ line_dicts[i]['parents']==['eSc_textblock_0f7e0cee','eSc_textblock_2cdb1f5e'] for i in range(4,6) ])
    assert line_dicts[6]['parents']==['eSc_textblock_1524f9d9','eSc_textblock_2cdb1f5e'] 

def test_region_extraction_regular( regular_dict):
    reg_dicts = sgf.region_dicts_from_segmentation_dict( regular_dict )
    assert len(reg_dicts)==5

def test_region_extraction_nested( nested_regions_dict):
    reg_dicts = sgf.region_dicts_from_segmentation_dict( nested_regions_dict )
    assert len(reg_dicts)==5
    assert [ len(reg['lines']) for reg in reg_dicts ] == [1, 3, 2, 1, 0] 

def test_flatten_segmentation_dict_regular( regular_dict):
    flat_dict = sgf.flatten_segmentation_dict( regular_dict )
    assert len(flat_dict['lines'])==7
    assert len(flat_dict['regions'])==5
    
def test_flatten_segmentation_dict_nested( nested_regions_dict):
    flat_dict = sgf.flatten_segmentation_dict( nested_regions_dict )
    assert len(flat_dict['lines'])==7
    assert len(flat_dict['regions'])==5

def test_alto_xml_to_page_xml( data_path ):
    """ Testing both the Alto → Page conversion and the validation function.
    """
    alto_path = data_path.joinpath('217_d9c7f_default.alto.xml')
    page_string = sgf.alto_to_page_xml_string( alto_path )
    assert sgf.page_xml_validate( page_string )
    
def test_page_xml_to_json( data_path ):
    page_xml_path = str(data_path.joinpath('217_d9c7f_default.page.xml'))
    segdict = sgf.segmentation_dict_from_page_xml( page_xml_path )
    assert sgf.json_validate( segdict )

def test_alto_xml_to_json_two_steps( data_path ):
    """ Testing both Alto → Page conversion and PageXML-as-string ingestion to JSON converter
    """
    alto_xml_path = str(data_path.joinpath('217_d9c7f_default.alto.xml'))
    page_xml_str = sgf.alto_to_page_xml_string( alto_xml_path )
    segdict = sgf.segmentation_dict_from_page_xml( page_xml_str )
    assert sgf.json_validate( segdict )

def test_alto_xml_to_json( data_path ):
    alto_xml_path = str(data_path.joinpath('217_d9c7f_default.alto.xml'))
    segdict = sgf.alto_to_segmentation_dict( alto_xml_path )
    assert sgf.json_validate( segdict ) 
