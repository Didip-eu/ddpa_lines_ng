"""
A module for:

+ conversions between established segmentation formats (ALTO, Page) and internal, DiDip-style JSON format
  (as documented by the corresponding schema in segformat_documents.py)
+ validation (Page or JSON)
+ transformations of a segmentation (JSON) dictionary: either grammatical (flattening, region and/or line extraction)
  or semantic (involving checking or changing the data)
+ ASCII visualization of a page segmentation.

"""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import lxml.etree as LET
import json
import re
import copy
import itertools
from io import StringIO
from enum import Enum
from datetime import datetime 
from typing import Callable, Optional, Union, Mapping, Any


import numpy as np
import shapely
import jsonschema
from unidecode import unidecode

from .segformat_documents import JsonSchema, XslAltoPage, PageXmlSchema

SegFormat = Enum('SegFormat', [('Unknown',0),('PAGE',1), ('ALTO',2), ('JSON',3)])


def get_format( segfile: str )->int:
    """
    Detect and return segmentation metadata format:
    + Page
    + Alto
    + JSON

    Args:
        segfile (str): segmentation file.
    Returns:
        Segformat: format code.
    """
    page_regexp = r'<([^:<>]+:)?PcGts.+xmlns'
    alto_regexp = r'<([^:<>]+:)?alto.+xmlns'
    with open(segfile) as segfile_if:
        try:
            current_line = segfile_if.readline()
        except UnicodeDecodeError as e:
            return SegFormat.Unknown
        # pass xml declaration and any empty subsequent line
        if re.match(r'<\?xml[^>]+>\s*$', current_line):
            while True:
                current_line = segfile_if.readline()
                if not re.match(r'^\s*$', current_line ):
                    break
        if re.search(alto_regexp, current_line):
            return SegFormat.ALTO
        elif re.search(page_regexp, current_line):
            return SegFormat.PAGE
        else:
            segfile_if.seek(0)
            try:
                segdict = json.load( segfile_if )
                # coarse check
                if 'image_filename' in segdict and 'regions' in segdict:
                    return SegFormat.JSON
                return SegFormat.Unknown
            except ValueError:
                print(f"{Path(__file__).name}.get_format: unknown (non-JSON) format!")
                return SegFormat.Unknown


def page_xml_from_segmentation_dict(seg_dict: str, pagexml_filename: str='', polygon_key='coords', with_text=False):
    """Serialize a JSON dictionary describing the lines into a PageXML file.
    Caution: this is a crude function, with no regard for validation.

    Args:
        seg_dict (dict[str,Union[str,list[Any]]]): segmentation dictionary of the form

            {"text_direction": ..., "type": "baselines", "lines": [{"tags": ..., "baseline": [ ... ]}]}
            or
            {"text_direction": ..., "type": "baselines", "regions": [ {"id": "r0", "lines": [{"tags": ..., "baseline": [ ... ]}]}, ... ]}
        pagexml_filename (str): if provided, output is saved in a PageXML file (standard output is the default).
        polygon_key (str): if the segmentation dictionary contain alternative polygons (f.i. 'extBoundary'),
            use them, instead of the usual line 'coords'.
        with_text (bool): encode line transcription, if it exists. Default is False.
    """
    def boundary_to_point_string( list_of_pts ):
        return ' '.join([ f"{pair[0]:.0f},{pair[1]:.0f}" for pair in list_of_pts ] )

    rootElt = ET.Element('PcGts', attrib={
        "xmlns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15", 
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance", 
        "xsi:schemaLocation": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd"})
    metadataElt = ET.SubElement(rootElt, 'Metadata')
    creatorElt = ET.SubElement( metadataElt, 'Creator')
    creatorElt.text=seg_dict['metadata']['creator'] if ('metadata' in seg_dict and 'creator' in seg_dict['metadata']) else 'Universität Graz/DH/nicolas.renet@uni-graz.at'
    createdElt = ET.SubElement( metadataElt, 'Created')
    createdElt.text=datetime.now().isoformat("T", "seconds")
    lastChangeElt = ET.SubElement( metadataElt, 'LastChange')
    lastChangeElt.text=createdElt.text
    commentElt = ET.SubElement( metadataElt, 'Comments')
    if 'comments' in seg_dict['metadata']:
        commentElt.text = seg_dict['metadata']['comments']
    # for back-compatibility
    elif 'comment' in seg_dict:
        commentElt.text = seg_dict['comment']
    if 'line_height_factor' in seg_dict:
        lineHeightFactorElt = ET.SubElement( metadataElt, 'LineHeightFactor' )
        lineHeightFactorElt.text = str(seg_dict['line_height_factor'])

    img_name = Path(seg_dict['image_filename']).name
    img_width, img_height = seg_dict['image_width'], seg_dict['image_height']    
    pageElt = ET.SubElement(rootElt, 'Page', attrib={'imageFilename': img_name, 'imageWidth': f"{img_width}", 'imageHeight': f"{img_height}"})
    # if no region in segmentation dict, create one (image-wide)
    if 'regions' not in seg_dict:
        seg_dict['regions']=[{'id': 'r0', 'coords': [[0,0],[img_width-1,0],[img_width-1,img_height-1],[0,img_height-1]]}, ]
    for reg in seg_dict['regions']:
        reg_xml_id = f"r{reg['id']}" if (type(reg['id']) is int or reg['id'][0]!='r') else reg['id']
        regElt = ET.SubElement( pageElt, 'TextRegion', attrib={'id': reg_xml_id})
        ET.SubElement(regElt, 'Coords', attrib={'points': boundary_to_point_string(reg['coords'])})
        # 3 cases: 
        # - top-level list of lines with region ref
        # - top-level list of lines with no regions
        # - top-level regions with a list of lines in each
        lines = [ l for l in seg_dict['lines'] if (('region' in l and l['region']==reg['id']) or 'region' not in l) ] if 'lines' in seg_dict else reg['lines']
        for line in lines:
            line_xml_id = f"l{line['id']}" if type(line['id']) is int else line['id']
            textLineElt = ET.SubElement( regElt, 'TextLine', attrib={'id': line_xml_id} )
            ET.SubElement( textLineElt, 'Coords', attrib={'points': boundary_to_point_string(line[polygon_key])} )
            if 'baseline' in line:
                ET.SubElement( textLineElt, 'Baseline', attrib={'points': boundary_to_point_string(line['baseline'])})
            if with_text and 'text' in line:
                ET.SubElement( ET.SubElement( textLineElt, 'TextEquiv'), 'Unicode').text = line['text']

    tree = ET.ElementTree( rootElt )
    ET.indent(tree, space='\t', level=0)
    if pagexml_filename:
        tree.write( pagexml_filename, encoding='utf-8' )
    else:
        tree.write( sys.stdout, encoding='unicode' )


def segmentation_dict_from_page_xml(page_source: str, get_text=True, regions_as_boxes=True, strict=False, region_line_overlap=.9) -> dict[str,Union[str,list[Any]]]:
    """Given a pageXML file name, return a JSON dictionary describing the lines.
    The resulting dictionary is flat, with two separate entries for lines and regions.

    Args:
        page_source (str): path of a PageXML file, or a PageXML string.
        get_text (bool): extract line text content, if present (default: False); this
            option causes line with no text to be yanked from the dictionary.
        regions_as_boxes (bool): when regions have more than 4 points or are not rectangular,
            store their bounding boxes instead; the boxe's boundary is determined
            by its pertaining lines, not by its nominal coordinates(default: True).
        strict (bool): if True, raise an exception if line coordinates are not comprised within
            their region's boundaries; otherwise (default), the region value is automatically
            extended to encompass the line coordinates.
        region_line_overlap (float): lines that do not overlap their region by this
            threshold are removed from the output dictionary.

    Returns:
        dict[str,Union[str,list[Any]]]: a dictionary of the form::

            {"metadata": { ... },
             "text_direction": ..., "type": "baselines", 
             "regions": [{"id": ..., "coords": [ ... ]}, 
                          "lines": [{"id": ..., "coords": [ ... ], "baseline": [ ... ]}, ... ]]
            }

           Regions are stored as a top-element.
    TODO:
        - check that unhandled exception on U-17_0995_s01.xml (AttributeError) has been fixed.
        - assign each line its containing region_s_ as a property.
    """
    def parse_coordinates( pts ):
        return [ [ int(p) for p in pt.split(',') ] for pt in pts.split(' ') ]

    def construct_line_entry(line: ET.Element, parent_region_ids: list = [] ) -> dict:
            line_id = line.get('id')
            baseline_elt = line.find('./pc:Baseline', ns)
            if baseline_elt is None:
                return None
            bl_points = baseline_elt.get('points')
            if bl_points is None or len(bl_points)==0:
                return None
            baseline_points = parse_coordinates( bl_points )
            coord_elt = line.find('./pc:Coords', ns)
            if coord_elt is None:
                return None
            c_points = coord_elt.get('points')
            if c_points is None or len(c_points)==0:
                return None
            polygon_points = parse_coordinates( c_points )

            line_text, line_custom_attribute = '', ''
            if get_text:
                text_elt = line.find('./pc:TextEquiv', ns) 
                if text_elt is not None:
                    line_custom_attribute = text_elt.get('custom') if 'custom' in text_elt.keys() else ''
                    unicode_elt = text_elt.find('./pc:Unicode', ns)
                    if unicode_elt is not None:
                        line_text = unicode_elt.text 
            line_dict = {'id': line_id, 'baseline': baseline_points, 
                        'coords': polygon_points }
            if line_text and not re.match(r'\s*$', line_text):
                line_dict['text'] = line_text 
                if line_custom_attribute:
                    line_dict['custom']=line_custom_attribute
            elif get_text:
                return None
            return line_dict

    def line_to_region_overlap(line_dict: dict, region_dict: dict):
        """ Check overlap between line's bbox and region boundaries."""
        line_bbox = shapely.envelope( shapely.multipoints( np.array( line_dict['coords'] )))
        reg_bbox = shapely.envelope( shapely.multipoints( np.array( region_dict['coords'] )))
        return reg_bbox.intersection( line_bbox ).area / line_bbox.area

    def extend_box( box_coords, inner_coords ):
        """Extend box coordinates to encompass inner boundaries """
        inner_xs, inner_ys = [ pt[0] for pt in inner_coords ], [ pt[1] for pt in inner_coords ]
        inner_left, inner_right, inner_top, inner_bottom = min(inner_xs), max(inner_xs), min(inner_ys), max(inner_ys)
        return [ [ inner_left if inner_left < box_coords[0][0] else box_coords[0][0],
                 inner_top if inner_top < box_coords[0][1] else box_coords[0][1]],
                [ inner_right if inner_right > box_coords[1][0] else box_coords[1][0],
                 inner_top if inner_top < box_coords[1][1] else box_coords[1][1]],
                [ inner_right if inner_right > box_coords[2][0] else box_coords[2][0],
                 inner_bottom if inner_bottom > box_coords[2][1] else box_coords[2][1]],
                [ inner_left if inner_left < box_coords[3][0] else box_coords[3][0],
                 inner_bottom if inner_bottom > box_coords[3][1] else box_coords[3][1]],]

    def process_region( region: ET.Element, region_accum: list, parent_region_ids:list ):
        # order of regions: inner -> outer
        parent_region_ids = [ region.get('id') ] + parent_region_ids
        region_coord_elt, rg_points = region.find('./pc:Coords', ns), None
        if region_coord_elt is not None:
            rg_points = region_coord_elt.get('points')
            if rg_points is None:
                raise ValueError("Region has no coordinates. Aborting.")
            rg_points = parse_coordinates( rg_points )
            if regions_as_boxes:
                xs, ys = [ pt[0] for pt in rg_points ], [ pt[1] for pt in rg_points ]
                left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
                rg_points = [[left,top], [right,top], [right,bottom], [left, bottom]]

        region_accum.append( {'id': region.get('id'), 'coords': rg_points, 'lines': [] } )

        for line_idx, elt in enumerate( list(region.iter())[1:] ):
            if elt.tag == "{{{}}}TextLine".format(ns['pc']):
                line_entry = construct_line_entry( elt )
                if line_entry is None:
                    continue
                overlap = line_to_region_overlap(line_entry, region_accum[-1] )
                if overlap < 0.5:
                    if strict:
                        raise ValueError("Page {}, region {}, l. {}: boundaries are not contained within its region. To disable this exception, pass strict=False".format(page, parent_region_ids[0], line_idx))
                    # extend region to fit the line
                    #elif overlap >= region_line_overlap:
                    #    region_accum[-1]['coords'] = extend_box( region_accum[-1]['coords'], line_entry['coords']+line_entry['baseline'] )
                    #else:
                    #    print(f"Line {line_entry['id']} does not meet overlap threshold with region ({overlap:.2f} < {region_line_overlap}): skipping.")
                    #    continue
                region_accum[-1]['lines'].append( line_entry )

            elif elt.tag == "{{{}}}TextRegion".format(ns['pc']):
                process_region(elt, region_accum, parent_region_ids)
    
    def extract_ns_from_file( f ):
        page_root, ns = None, {}
        with open( f, 'r' ) as page_file:
            for line in page_file:
                m = re.search(r'xmlns="([^"]+)"', line)
                if m:
                    ns['pc'] = m.group(1)
                    page_file.seek(0)
                    break
            page_root = ET.parse( page_file ).getroot()
        return page_root, ns

    # if source is an XML string
    page_root, ns = None, {}
    if isinstance( page_source, Path):
        page_root, ns = extract_ns_from_file( page_source )
    else:
        try:
            page_root = ET.fromstring( page_source )
            ns['pc'] = re.search(r'xmlns="([^"]+)"', page_source).group(1)
        except (ET.ParseError) as e:
            page_root, ns = extract_ns_from_file( page_source )

    if page_root is None:
        raise ValueError("Could not parse the source. Abort.")
    if 'pc' not in ns:
        raise ValueError(f"Could not find a name space in file {page_root}. Parsing aborted.")

    regions, page_dict = [], {}
    metadata_elt = page_root.find('./pc:Metadata', ns)
    creation_timestamp = datetime.now().isoformat('T', 'seconds')
    if metadata_elt is None:
        page_dict = { 'metadata': { 'created': creation_timestamp, 'creator': Path(__file__).name, } }
    else:
        created_elt, creator_elt = [ metadata_elt.find(f"./pc:{tag}", ns) for tag in ('Created', 'Creator') ]
        page_dict = {
                'metadata': {
                    'created': created_elt.text if created_elt else creation_timestamp,
                    'creator': creator_elt.text if creator_elt else Path(__file__).name,
                    'comments': f"Converted from PageXML file {page_source} with {Path(__file__).name}.",
                }
        }
    page_dict['type']='baselines'
    page_dict['text_direction']='horizontal-lr'

    pageElement = page_root.find('./pc:Page', ns)
    
    page_dict['image_filename']=pageElement.get('imageFilename')
    page_dict['image_width'], page_dict['image_height']=[ int(pageElement.get('imageWidth')), int(pageElement.get('imageHeight'))]

    for textRegionElement in pageElement.findall('./pc:TextRegion', ns):
        process_region( textRegionElement, regions, [] )

    page_dict['regions'] = regions

    return page_dict 


def segdict_sink_lines_deprecate(segdict: dict):
    """Convert a segmentation dictionary with top-level line array ('lines') 
    to a nested dictionary where each region in the 'regions' array contains its 
    corresponding 'lines' array. No change applied if lines are already wrapped
    into the regions.
    NOTE: TO BE DEPRECATED.

    Args:
        segdict (dict): segmentation dictionary of the form::

                {..., "lines": [ {"id":..., "regions": [...]}, ... ], "regions": [ ... ] }

            OR

                {..., "lines": [ {"id":..., "region": "r0"}, ... ], "regions": [ ... ] }

    Returns:
        dict: a modified copy of the original dictionary::

            {..., "regions": [ {"id":..., lines=[{"id": ... }, ... ]}, ... ] }
    """
    segdict = copy.deepcopy(segdict)
    if 'lines' not in segdict or not segdict['lines']:
        return segdict
    # if no 'regions' entry for lines, assign to each line its proper region
    if 'regions' not in segdict['lines'][0]:
        for line in segdict['lines']:
            if 'region' in line:
                line['regions']=[ line['region'] ]
                del line['region']
            else:
                for reg in segdict['regions']:
                    if (line['coords'] >= np.min( reg['coords'], axis=0 )).all() and (line['coords'] <= np.max( reg['coords'], axis=0 )).all():
                        #print("Check coordinates")
                        if 'regions' not in line:
                            line['regions']=[]
                        line['regions'].append( reg['id'] )
    # fix old Kraken format
    if type(segdict['regions']) is dict:
        segdict['regions'] = segdict['regions']['text']

    for line in segdict['lines']:
        this_reg=[ reg for reg in segdict['regions'] if reg['id']==line['regions'][0] ][0] if ('regions' in line and line['regions']) else line['region']
        if 'lines' not in this_reg:
            this_reg['lines']=[]
        this_reg['lines'].append(line)
        del line['regions']
    del segdict['lines']

    # regions with no lines assigned are still valid
    for reg in segdict['regions']:
        if 'lines' not in reg:
            reg['lines']=[]
    return segdict

def alto_to_page_xml_string( segfile: str, xslfile=None)->str:
    """
    ALTO → Page conversion tool with embedded XSL stylesheet.

    Args:
        segfile (str): segmentation data (ALTO format)
        xslfile (str): XSL stylesheet (optional: use built-in stylesheet as a fallback).

    Returns:
        str: transformed XML output as a string.
    """

    if not Path(segfile).exists():
        raise FileNotFoundError(f"Could not find segmentation file {segfile}. Abort.")
    this_format = get_format( segfile )
    if this_format != SegFormat.ALTO:
        raise ValueError(f"Input file is not in Alto format (found: {this_format}). Abort.")
    source_file = segfile

    ns={'alto': "http://www.loc.gov/standards/alto/ns-v4#"}
    dom = LET.parse(source_file)

    # first, make polygon/line coordinates palatable for XSLT
    def coords_to_pairs( coord_str ):
        coord_str = re.sub(r'\s+',' ',coord_str.strip())
        coord = coord_str.split(' ')
        if len(coord)%2:
            raise(ValueError("Even number of coords expected! Abort."))
        pairs = ' '.join([ f"{coord[i]},{coord[i+1]}" for i in range(0, len(coord), 2) ])
        return pairs

    root = dom.getroot()
    for plg in root.findall('.//alto:Polygon', ns):
        points_str = plg.get('POINTS')
        plg.set('POINTS', coords_to_pairs( points_str ))
    for tl in root.findall('.//alto:TextLine', ns):
        points_str = tl.get('BASELINE')
        tl.set('BASELINE', coords_to_pairs( points_str ))

    # XSL transform
    transform = LET.XSLT( LET.parse( xslfile )) if xslfile and Path(xslfile).exists() else LET.XSLT( LET.XML( XslAltoPage.encode() ))
    newdom = transform(dom, today=LET.XSLT.strparam(datetime.now().isoformat("T","seconds")), source=LET.XSLT.strparam(Path(source_file).name))

    LET.indent( newdom, space='\t', level=0)
    return LET.tostring( newdom ).decode()

def alto_to_page_xml( segfile: str, xslfile=None, pagexml_filename: str='', overwrite_existing=True):
    """
    ALTO → Page conversion tool with embedded XSL stylesheet.

    Args:
        segfile (str): segmentation data (ALTO format)
        xslfile (str): XSL stylesheet (optional: use built-in stylesheet as a fallback).
1G      pagexml_filename (str): if provided, output is saved in a PageXML file (standard output is the default).
    """

    xml_str = alto_to_page_xml_string( segfile, xslfile=xslfile )
    if pagexml_filename and (not Path(pagexml_filename).exists() or overwrite_existing):
        with open( pagexml_filename, 'w') as pagexml_of:
            pagexml_of.write('<?xml version="1.0" encoding="utf-8"?>\n')
            pagexml_of.write( xml_str )
    else:
       print('<?xml version="1.0" encoding="utf-8"?>\n{}'.format(xml_str))
     

def alto_to_segmentation_dict( segfile: str )->dict:
    """
    ALTO → DiDip-style JSON dictionary.

    Args:
        segfile (str): segmentation data (ALTO format)
    Returns:
        dict: a segmentation dictionary.
    """
    return segmentation_dict_from_page_xml( alto_to_page_xml_string( segfile ))

def json_validate( segdict: dict, schema_dict=None)->bool:
    """
    Validate the dictionary against the given schema

    Args:
        segdict (dict): a DiDip-style segmentation dictionary.
        schema_dict (dict): a schema dictionary.

    Returns:
        bool: 1 if successful validation; 0 otherwise.
    """
    schema = schema_dict if schema_dict else JsonSchema
    try:
        jsonschema.validate( instance=segdict, schema=JsonSchema )
    except (jsonschema.exceptions.ValidationError, jsonschema.exceptions.SchemaError) as e:
        print(e)
        return 0
    return 1
     

def page_xml_validate( page_source: str, schema_source: str=PageXmlSchema ):
    """
    Validate a Page XML file, from a string or file.
    See `segformat_documents.py` module for Schema spec.

    Args:
        page_source (str): either a file path or an XML string.
        schema_file = 

    Returns:
        bool: True for successful validation; 0 otherwise.
    """
    xmlschema = None
    try:
        xmlschema=LET.XMLSchema( LET.parse( StringIO( schema_source )))
    except (LET.ParseError, LET.XMLSyntaxError) as e:
        if Path(schema_source).exists():
            xmlschema = LET.parse( schema_source )
    if not xmlschema:
        print("Could not parse a schema.")
        return False
    try:
        page_root = LET.fromstring( page_source )
        print(f"Read from string (type={type(page_source)}")
    except (LET.ParseError, LET.XMLSyntaxError) as e:
        page_root = LET.parse( page_source )
        print(f"Read from file {page_source}")

    return xmlschema.validate( page_root )

def json_doctor( segdict: dict, operations={'region_fit': True, 'line_surgery': True}, verbose=False )->dict:
    """
    Fix semantic issues in a JSON segmentation dictionary:

    + re-assign lines to their proper regions
    + extend regions to encompass their line bounding boxes
    
    Args:
        segdict (dict): segmentation dictionary, DiDip-style.

    Returns:
        dict: a modified copy of the input dictionary.
    """
    # Steps:
    # 1. Clip all coordinates to image size
    # 2. Line reassignment: fix most glaring errors regarding line-to-region assignments (tolerance
    #     for line region overlap is a parameter)
    # 3. Fine-tuning: regions can be extended to fit their line coordinates (overlap between regions
    #    is not considered an issue in most contexts)

    def extend_box( outer_coords, inner_coords ):
        """
        Extend outer box to fit the inner coordinates, within the image's limits.
        """
        outer_coords, inner_coords = np.array( outer_coords ), np.array( inner_coords )
        o_l, o_t, o_r, o_b = *(outer_coords.min(axis=0).tolist()), *(outer_coords.max(axis=0).tolist())
        i_l, i_t, i_r, i_b = *(inner_coords.min(axis=0).tolist()), *(inner_coords.max(axis=0).tolist())

        l = i_l if o_l >= i_l else o_l
        t = i_t if o_t >= i_t else o_t
        r = i_r if o_r <= i_r else o_r
        b = i_b if o_b <= i_b else o_b
        
        if r >= segdict['image_width']:
            r = segdict['image_width']-1
        if b >= segdict['image_height']:
            b = segdict['image_height']-1
        #if verbose:
        #    print(f"extended region: {[[l,t],[r,t],[r,b],[l,b]]}")
        return [[l,t],[r,t],[r,b],[l,b]]

    segdict_new = copy.deepcopy( segdict )
    # ensure that every line polygon is within image's limits
    if verbose:
        print("Check polygons against image limit...")
    for r in segdict_new['regions']:
        for l in r['lines']:
            new_coords=np.clip( l['coords'], [0,0], [segdict['image_width']-1, segdict['image_height']-1] ).tolist()
            if verbose and new_coords != l['coords']:
                print(f"region  {r['id']}, line {l['id']}: coords → {new_coords}")
            l['coords']=new_coords

    if verbose:
        print("Re-assign lines...")
    segdict_butchered = segdict_reassign_lines( segdict_new, verbose=verbose )
    if verbose and segdict_butchered != segdict_new:
        print("Some lines were reassigned!")
    if verbose:
        print("Extend regions around their lines...")
    for reg in segdict_butchered['regions']:
        inner_line_coords = [ c for l in reg['lines'] for c in l['coords']]
        if not inner_line_coords:
            continue
        new_coords = extend_box( reg['coords'], inner_line_coords )
        if verbose and new_coords != reg['coords']:
            print(f"region  {reg['id']} extended: {reg['coords']} → {new_coords}")
        reg['coords']=new_coords
    return segdict_butchered


def segdict_reassign_lines( segdict: dict, verbose=False):
    """
    Given a segmentation dictionary, reassign lines to their most likely containing regions:
    assign each line to region with maximum overlap, as a ratio of the line's area; between
    two competing regions, choose the smaller one.

    Args:
        segdict (dict): a segmentation dictionary, DiDip-style, with regions a top-level element.

    Returns:
        dict: a restructured dictionary.

    """
    def line_to_region_overlap(line_dict: dict, region_dict: dict):
        """ Check overlap between line's bbox and region boundaries."""
        line_bbox = shapely.envelope( shapely.multipoints( np.array( line_dict['coords'] )))
        reg_bbox = shapely.envelope( shapely.multipoints( np.array( region_dict['coords'] )))
        return reg_bbox.intersection( inner_plg ).area / line_bbox.area

    region_to_bbox = [ shapely.envelope( shapely.multipoints( np.array( r['coords'] ))) for r in segdict['regions'] ]
    new_segdict = copy.deepcopy( segdict )
    for r in new_segdict['regions']:
        r['lines']=[]
    # all lines, sorted by centroids
    lines = [ l for r in segdict['regions'] for l in r['lines'] ]  
    for l in lines:
        l['bbox']=shapely.envelope( shapely.multipoints( np.array( l['coords'] )))
        #print(l['bbox'])
    lines.sort( key=lambda ln: ln['bbox'].centroid.x )
    # map line index to (<region index>, overlap)
    line_to_region = [(-1,0.0) for l in lines ]
    #print(line_to_region)
    for l_idx, l in enumerate(lines):
        max_overlap = 0
        for r_idx, r in enumerate( segdict['regions'] ):
            this_overlap = region_to_bbox[r_idx].intersection( l['bbox'] ).area / l['bbox'].area
            #print(f"intersection: {region_to_bbox[r_idx].intersection( l['bbox'] ).area}", end=", ")
            #print(f"line box area: {l['bbox'].area}")
            #print(f"line {l['id']} ({l['bbox']}) / region: {r['id']} ({region_to_bbox[r_idx]}): overlap={this_overlap}")
            if this_overlap > max_overlap:
                max_overlap = this_overlap
                line_to_region[l_idx]=(r_idx, this_overlap )
                #if verbose:
                #    print(f"asssign line {l['id']} to region {r['id']}: overlap={this_overlap}")
            elif this_overlap == max_overlap and line_to_region[l_idx][0]>=0:
                stored_region_idx = line_to_region[l_idx][0] # region index
                if region_to_bbox[r_idx].area < region_to_bbox[stored_region_idx].area:
                    #if verbose:
                    #    print(f"asssign line {l['id']} to smaller region {r['id']}: overlap={this_overlap}")
                    line_to_region[l_idx]=(r_idx, this_overlap )
    # assign each line to its region object
    # (vertical sorting by centroid has been done previously)
    for l_idx, lr in enumerate( line_to_region ):
        del lines[l_idx]['bbox']
        new_segdict['regions'][ lr[0] ]['lines'].append( lines[l_idx] )
    if verbose: 
        for r in new_segdict['regions']:
            r_id, r_l, r_t, r_r, r_b = r['id'].replace('eSc_textblock_',''), *(np.array(r['coords']).min(axis=0).tolist()), *(np.array(r['coords']).max(axis=0).tolist())
            print(f"region {r_id}: [<{r_l},{r_r}>, <{r_t}, {r_b}>]")
            region_bboxes=[ (l['id'].replace('eSc_line_',''), *(np.array(l['coords']).min(axis=0).tolist()), *(np.array(l['coords']).max(axis=0).tolist()) ) for l in r['lines'] ]
            for  l_id, l_l, l_t, l_r, l_b in sorted( region_bboxes, key=lambda x: x[2] ):
                print(f"\tline {l_id}: [<{l_l},{l_r}>, <{l_t},{l_b}>]")

    return new_segdict


def flatten_segmentation_dict( segmentation_dict: dict ) -> dict:
    """
    Flatten a DiDip-style segmentation dictionary, with both lines and regions stored as top-level lists.

    Args:
        segmentation_dict (dict): a dictionary, typically constructed from a JSON file.
    Returns:
        dict: segmentation dictionary of the form::

            TBD
    """
    regions = region_dicts_from_segmentation_dict( segmentation_dict )
    lines = line_dicts_from_segmentation_dict( segmentation_dict )
    for r in regions:
        if 'lines' in r:
            del r['lines']
    return {
            'metadata': segmentation_dict['metadata'],
            'regions': regions,
            'lines': lines,
    }


def line_dicts_from_segmentation_dict( segmentation_dict: dict ) -> list[dict]:
    """From a segmentation dictionary, return a list of all line dictionaries.

    Args:
        segmentation_dict (dict): a dictionary, typically constructed from a JSON file. The 'lines' entry is either
        top-level key, or nested in a region or subregion.
    Returns:
        list[dict]: a list of dictionaries; each line stores the id(s) of its containing region(s), the innermost first.
    """
    def get_lines_from_region( reg, parent_region_stack ):
        subregion_lines = []
        if 'lines' in reg:
            for l in reg['lines']:
                l['parents']=parent_region_stack
        subregion_lines = list(itertools.chain.from_iterable([ get_lines_from_region( inner_reg, [inner_reg['id']] + parent_region_stack ) for inner_reg in reg['regions']])) if 'regions' in reg else []
        return (reg['lines'] if 'lines' in reg else []) + subregion_lines
    return get_lines_from_region( copy.deepcopy(segmentation_dict), [] )


def region_dicts_from_segmentation_dict( segmentation_dict: dict ) -> list[dict]:
    """From a segmentation dictionary, return a flat list of all (possibly nested) regions."

    Args:
        segmentation_dict (dict): a dictionary, typically constructed from a JSON file.
    Returns:
        list[dict]: a list of region dictionaries.
    """
    def get_regions( reg, reg_accum ):
        if reg != segmentation_dict:
            reg_accum.append( reg )
        if 'regions' not in reg:
            return []
        for inner_reg in reg['regions']:
            get_regions( inner_reg, reg_accum )
    regions = []
    get_regions( copy.deepcopy(segmentation_dict), regions )
    for r in regions:
        if 'regions' in r:
            del r['regions']
    return regions


def any_to_ascii( segfile: str, scale_hw=(.01,.02), lines=0, repair=False, text=False)->str:
    """
    ASCII-rendition of a segmentation file.

    Args:
        segfile (str): path of a JSON, Page, or Alto segmentation file.
        scale_hw (Union[tuple[float,float],float,int]): if passed a tuple, interpreted as scaling
            factor for pixel-to-terminal-line and pixel-to-terminal-col (respectively) coordinate 
            transformation; if passed a single number between .5 and 5, interpreted as a factor of
            the default scale (=.01,.02).
        lines (int): if non-zeero, show line ids within their regions: 1=only lines that fit within
            region display box; 2=lines that fit within canvas display box.
        repair (bool): try repair a faulty segmentation (default: False).
        text (bool): display text, if present (default: False).

    Returns:
        str: a character-based rendition of the layout.
    """
    segdict = None
    segmentation_format = get_format( segfile )
    if segmentation_format == SegFormat.Unknown:
        print(f"{Path(__file__).name}.any_to_ascii: Could not determine input format. Abort.")
        return ''
    if segmentation_format == SegFormat.JSON:
        with open(segfile) as seg_if:
            segdict = json.load( seg_if )
    elif segmentation_format == SegFormat.PAGE:
        segdict = segmentation_dict_from_page_xml( segfile )
    elif segmentation_format == SegFormat.ALTO:
        segdict = segmentation_dict_from_page_xml( alto_to_page_xml_string( segfile ))

    if not segdict:
        raise ValueError("Could not parse a valid segmentation dictionary. Abort.")

    if repair:
        segdict = json_doctor( segdict )
    return segdict_to_ascii( segdict, scale_hw=scale_hw, lines=lines, text=text)

def segdict_to_ascii( segdict:dict, scale_hw=(.01,.02), lines=0, summary=True, text=False)->str:
    """
    ASCII-rendition of a JSON segmentation dictionary.

    Eg.::

        ┼┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┌────────────────┐┄┄┄┄┄┄┼
        ┆                  │r:2a93943e      │      ┆
        ┆     ┌────────────└──l:8┌────────────────┐┆
        ┆     │r:ef1657c4       ││r:ed8f3022      │┆
        ┆     │  l:1e355b0a...  ││  l:8590f3bf... │┆
        ┆     │  l:1e1c59e1.... ││  l:5c1a7591... │┆
        ┆     │  l:aa302e79...  ││  l:50a89193... │┆
        ┆     │  l:0bcc2987...  ││  l:ab831781... │┆
        ┆     │  l:f45a1fa8...  ││  l:b808a65a... │┆
        ┆     │  l:1d001e4b...  ││  l:640bb0ff....│┆
        ┆     │  l:e30c7a81..   │┌────┐0a959ab... │┆
        ┆     │  l:7bb2bb1f.... ││r:8fa3e8012b... │┆
        ┆     │  l:85832788...  ││  l:│7383a84... │┆
        ┆     ┌──────┐d5625...  │└────┘bd94110..  │┆
        ┆     │r:f77b369130..   ││  l:2b0c22bb... │┆
        ┆     │  l:a6│ecf48...  ││  l:1dac6035... │┆
        ┆     │  l:6f│ff3a2...  ││  l:acb1c73e... │┆
        ┆     └──────┘60b9c...  ││  l:5393c17b... │┆
        ┆     │  l:299e800c...  ││  l:4e171c9c... │┆
        ┆     │  l:508f7f75...  ││  l:d1ffc59c... │┆
        ┆     │  l:496470a0...  ││  l:5c56470d... │┆
        ┆     │  l:0ba0e73a...  ││  l:8a93cee6    │┆
        ┆     │  l:30b7e6ef...  ││  l:6d2db756    │┆
        ┆     │  l:e66fbbbb...  ││  l:b48006e0... │┆
        ┆     │  ...            ││  ...           │┆
        ┆     └─────────────────┘└────────────────┘┆
        ┆                                          ┆
        ┼┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┼

    Args:
        segdict (dict): path of a JSON segmentation file.
        scale_hw (Union[tuple[float,float],float,int]): if passed a tuple, interpreted as scaling
            factor for pixel-to-terminal-line and pixel-to-terminal-col (respectively) coordinate 
            transformation; if passed a single number between .5 and 5, interpreted as a factor of
            the default scale (=.01,.02).
        lines (int): if non-zeero, show line ids within their regions: 1=only lines that fit within
            region display box; 2=lines that fit within canvas display box.
        summary (bool): print a short summy of the layout features (image size, regions and lines)

    Returns:
        str: an terminal-friendly representation of the layout.
    """

    def longest_common_prefix( words ):
        if not words:
            return ''
        max_length = sorted( [ len(w) for w in words ] )[0]
        for i in range(max_length):
            if not all( [ words[0][i]==words[j][i] for j in range(len(words)) ] ):
                break
        return words[0][:i]

    if not segdict:
        raise ValueError("Provide a valid segmentation dictionary, or a segmentation file.")
        
    default_scale = (.01,.02)
    width, height = segdict['image_width'], segdict['image_height']
    if type(scale_hw) in [float,int]:
        if scale_hw < .5:
            print("Warning: scaling factor to be applied to the default display ratio set to lowest permissible value (0.5)")
            scale_hw = .5
        elif scale_hw > 3.0:
            print("Warning: scaling factor to be applied to the default display ratio set to highest permissible value (3.0)")
            scale_hw = 3.0
        scale_hw=[ s*scale_hw for s in default_scale ] 
    canvas = np.full( (np.array([height,width])*scale_hw).astype('uint16'), ord(' '))
    # page box
    canvas[[0,0,-1,-1],[0,-1,-1,0]]=ord('┼')
    canvas[1:-1,[0,-1]]=ord('┆')
    canvas[[0,-1],1:-1]=ord('┄')
    reg_id_prefix = longest_common_prefix( [ reg['id'] for reg in segdict['regions']] )
    line_ids = [ l['id'] for reg in segdict['regions'] if 'lines' in reg for l in reg['lines']]
    line_id_prefix = longest_common_prefix( line_ids ) if lines else ''
    for reg in segdict['regions'][:]:
        reg['id']=reg['id'].replace( reg_id_prefix, 'r:')
        reg_arr = np.array( [c[::-1] for c in reg['coords']] ).astype('uint16')
        lt, rb = reg_arr[:,::-1].min(axis=0).tolist(), reg_arr[:,::-1].max(axis=0).tolist()
        scaled_reg_hw = (rb[0]-lt[0]+1, rb[1]-lt[1]+1)
        #print(*(np.array(canvas.shape)-1, (reg_arr*scale_hw).astype('uint16').T))
        scaled_reg_arr = np.clip((reg_arr*scale_hw).astype('uint16').T, None, [[canvas.shape[0]-1],[canvas.shape[1]-1]])
        # boxes
        canvas[ scaled_reg_arr[0], scaled_reg_arr[1] ]=ord('+')
        canvas[ scaled_reg_arr[0,0:4], scaled_reg_arr[1,0:4] ]=[ ord(c) for c in '┌┐┘└']
        canvas[ scaled_reg_arr[0,0]+1:scaled_reg_arr[0,2], scaled_reg_arr[1,[0,1]]]=ord('│')
        canvas[ scaled_reg_arr[0,[1,2]], scaled_reg_arr[1,0]+1:scaled_reg_arr[1,2]]=ord('─')

        # region ids
        reg_id_as_intlist = [ ord(c) for c in reg['id'] ]
        region_id_offset = 1
        reg_id_start_x = int(scaled_reg_arr[1,0]+region_id_offset)
        reg_id_end_x = reg_id_start_x + len( reg['id'] )
        reg_id_cut = max(0, reg_id_end_x-canvas.shape[1]) # region's id too long to fit into the canvas
        # discard degenerate region on the canvas lower edge 
        if scaled_reg_arr[0,0] < canvas.shape[0]-1:
            canvas[ scaled_reg_arr[0,0]+1, reg_id_start_x:reg_id_end_x ]=reg_id_as_intlist[0:len(reg_id_as_intlist)-reg_id_cut]
        # lines
        if lines and 'lines' in reg:
            sorted_lines = sorted(reg['lines'], key=lambda x: x['coords'][0][1])
            for i,l in enumerate([l['id'].replace(line_id_prefix,'l:') for l in sorted_lines]):
                l_xs =np.array( sorted_lines[i]['coords'] )[:,0]
                line_display_length = int(np.floor((l_xs.max()-l_xs.min()+1) * scale_hw[1] ))-1
                l_id_as_intlist = [ ord(c) for c in l ]
                line_id_offset = region_id_offset+2
                canvas_row_idx, canvas_col_idx = scaled_reg_arr[0,0]+2+i, scaled_reg_arr[1,0]+line_id_offset
                # omitting lines beyond the canvas' size
                if len(reg['lines'])!=1 and (lines==1 and canvas_row_idx >= scaled_reg_arr[0,2]-1) or (lines == 2 and canvas_row_idx >= canvas.shape[0]-2): 
                    #canvas[ canvas_row_idx, canvas_col_idx:canvas_col_idx+len('...')] = [ ord(c) for c in '...' ]
                    break
                else:
                    line_end_x = int(canvas_col_idx+line_display_length)
                    line_cut = max(0, line_end_x - int(scaled_reg_arr[1,2]))
                    line_display_length -= line_cut
                    canvas[ canvas_row_idx, canvas_col_idx:canvas_col_idx+line_display_length ] = [ ord('.') ] * line_display_length 
                    # display text content instead of id
                    line_content = l_id_as_intlist
                    if text and 'text' in sorted_lines[i]:
                        line_content = [ ord(c) for c in unidecode(sorted_lines[i]['text']) ]
                    line_content_end_x = int(canvas_col_idx+len(line_content))
                    line_content_cut = max(0, line_content_end_x - int(scaled_reg_arr[1,2] ))
                    line_display_length = max(1, len(line_content) - line_content_cut)
                    #print(f"canvas[{canvas_col_idx}:{canvas_col_idx+id_display_length}")
                    canvas[ canvas_row_idx, canvas_col_idx:canvas_col_idx+line_display_length] = line_content[:line_display_length]
                    #id_end_x -= id_cut
                    #canvas[ canvas_row_idx, canvas_col_idx:id_end_x] = l_id_as_intlist[:len(l_id_as_intlist)-id_cut] 

    # Summary: sort regions by area size
    lines_per_region = '\n'.join([ f"  {reg['id']}: {len(reg['lines'])} l." 
                                  for reg in sorted( segdict['regions'], 
                                                    key=lambda r: (r['coords'][1][0]-r['coords'][0][0])*(r['coords'][2][1]-r['coords'][1][1]), reverse=True )])
    summary_text = "\n".join([ 
                f"Image filename: {segdict['image_filename']}",
                f"Image size: {segdict['image_width']} x {segdict['image_height']}",
                f"Lines: {len(line_ids)}",
                f"Regions: {len(segdict['regions'])}",
                lines_per_region
                              ]) if summary else ''

    page_display = '\n'.join([(''.join([ chr(c) for c in l ])) for l in canvas ] )

    return f"\n{summary_text}\n\n{page_display}" 

