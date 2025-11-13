#!/usr/bin/env python3
"""
A stand-alone script for PageXML -> JSON conversion.

The original function is in ddpa_lines_ng/libs/seglib.
"""

import sys
import json
import fargv
import re
from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Union, Any



p = {
    'file_paths': set([]),
    'output_format': ('json', 'stdout'),
    'get_text': [1, "Extract text content of the line, if it exists"],
    "comment": ['',"A text string to be added to the <Comments> elt."],
}


def segmentation_dict_from_xml(page: str, get_text=False, regions_as_boxes=True, strict=False) -> dict[str,Union[str,list[Any]]]:
    """Given a pageXML file name, return a JSON dictionary describing the lines.

    Args:
        page (str): path of a PageXML file.
        get_text (bool): extract line text content, if present (default: False); this
            option causes line with no text to be yanked from the dictionary.
        regions_as_boxes (bool): when regions have more than 4 points or are not rectangular,
            store their bounding boxes instead; the boxe's boundary is determined
            by its pertaining lines, not by its nominal coordinates(default: True).
        strict (bool): if True, raise an exception if line coordinates are comprised within
            their region's boundaries; otherwise (default), the region value is automatically
            extended to encompass the line coordinates.

    Returns:
        dict[str,Union[str,list[Any]]]: a dictionary of the form::

            {"metadata": { ... },
             "text_direction": ..., "type": "baselines", 
             "lines": [{"id": ..., "coords": [ ... ], "baseline": [ ... ]}, ... ],
             "regions": [{"id": ..., "coords": [ ... ]}, ... ] }

           Regions are stored as a top-element.

    """
    def parse_coordinates( pts ):
        return [ [ int(p) for p in pt.split(',') ] for pt in pts.split(' ') ]

    def construct_line_entry(line: ET.Element, region_ids: list = [] ) -> dict:
            line_id = line.get('id')
            baseline_elt = line.find('./pc:Baseline', ns)
            if baseline_elt is None:
                return None
            bl_points = baseline_elt.get('points')
            if bl_points is None:
                return None
            baseline_points = parse_coordinates( bl_points )
            coord_elt = line.find('./pc:Coords', ns)
            if coord_elt is None:
                return None
            c_points = coord_elt.get('points')
            if c_points is None:
                return None
            polygon_points = parse_coordinates( c_points )

            line_text, line_custom_attribute = '', ''
            if get_text:
                text_elt = line.find('./pc:TextEquiv', ns)
                if text_elt is None:
                    return None
                line_custom_attribute = text_elt.get('custom') if 'custom' in text_elt.keys() else ''
                unicode_elt = text_elt.find('./pc:Unicode', ns)
                if unicode_elt is not None:
                    line_text = unicode_elt.text
            line_dict = {'id': line_id, 'baseline': baseline_points,
                        'coords': polygon_points, 'regions': region_ids}
            if line_text and not re.match(r'\s*$', line_text):
                line_dict['text'] = line_text
                if line_custom_attribute:
                    line_dict['custom']=line_custom_attribute
            elif get_text:
                return None
            return line_dict

    def check_line_entry(line_dict: dict, region_dict: dict):
        """ Check whether line coords are within region boundaries."""
        reg_l, reg_t, reg_r, reg_b = region_dict['coords']
        return all([ (pt[0] >= reg_l[0] and pt[0] <= reg_r[0] and pt[1] >= reg_t[1] and pt[1] <= reg_b[1]) for pt in line_dict['coords']])

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

    def process_region( region: ET.Element, region_accum: list, line_accum: list, region_ids:list ):
        # order of regions: outer -> inner
        region_ids = region_ids + [ region.get('id') ]

        region_coord_elt = region.find('./pc:Coords', ns)
        if region_coord_elt is not None:
            rg_points = region_coord_elt.get('points')
            if rg_points is None:
                raise ValueError("Region has no coordinates. Aborting.")
            rg_points = parse_coordinates( rg_points )
            if regions_as_boxes:
                xs, ys = [ pt[0] for pt in rg_points ], [ pt[1] for pt in rg_points ]
                left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
                rg_points = [[left,top], [right,top], [right,bottom], [left, bottom]]

        region_accum.append( {'id': region.get('id'), 'coords': rg_points } )

        for line_idx, elt in enumerate( list(region.iter())[1:] ):
            if elt.tag == "{{{}}}TextLine".format(ns['pc']):
                line_entry = construct_line_entry( elt, region_ids )
                if line_entry is None:
                    continue
                if not check_line_entry(line_entry, region_accum[-1] ):
                    if strict:
                        raise ValueError("Page {}, region {}, l. {}: boundaries are not contained within its region.".format(page, region_ids[-1], line_idx))
                    else: # extend region's bounding box boundary
                        region_accum[-1]['coords'] = extend_box( region_accum[-1]['coords'], line_entry['coords']+line_entry['baseline'] )
                line_accum.append( line_entry )
            elif elt.tag == "{{{}}}TextRegion".format(ns['pc']):
                process_region(elt, region_accum, line_accum, region_ids)

    with open( page, 'r' ) as page_file:

        # extract namespace
        ns = {}
        for line in page_file:
            m = re.match(r'\s*<([^:]+:)?PcGts\s+xmlns(:[^=]+)?=[\'"]([^"]+)["\']', line)
            if m:
                ns['pc'] = m.group(3)
                page_file.seek(0)
                break

        if 'pc' not in ns:
            raise ValueError(f"Could not find a name space in file {page}. Parsing aborted.")

        lines = []
        regions = []
        page_dict = {}

        page_tree = ET.parse( page_file )
        page_root = page_tree.getroot()

        metadata_elt = page_root.find('./pc:Metadata', ns)
        if metadata_elt is None:
            page_dict = { 'metadata': { 'created': str(datetime.now()), 'creator': __file__, } }
        else:
            created_elt = metadata_elt.find('./pc:Created', ns)
            creator_elt = metadata_elt.find('./pc:Creator', ns)
            comments_elt = metadata_elt.find('./pc:Comments', ns)
            page_dict: {
                    'metadata': {
                        'created': created_elt.text if created_elt else str(datetime.datetime.now()),
                        'creator': creator_elt.text if creator_elt else __filename__,
                        'comments': comments_elt.text if comments_elt else "",
                    }
            }

        page_dict['type']='baselines'
        page_dict['text_direction']='horizontal-lr'

        pageElement = page_root.find('./pc:Page', ns)

        page_dict['image_filename']=pageElement.get('imageFilename')
        page_dict['image_width'], page_dict['image_height']=[ int(pageElement.get('imageWidth')), int(pageElement.get('imageHeight'))]


        for textRegionElement in pageElement.findall('./pc:TextRegion', ns):
            process_region( textRegionElement, regions, lines, [] )

        page_dict['lines'] = lines
        page_dict['regions'] = regions

    return page_dict


if __name__ == '__main__':

    args, _ = fargv.fargv( p )

    for xml_path in args.file_paths:

        xml_path = Path(xml_path)

        segdict = segmentation_dict_from_xml( xml_path, get_text=args.get_text )
        segdict_str = json.dumps( segdict, indent=4 )

        if args.output_format == 'stdout':
            print( segdict_str )
        else:
            json_path = xml_path.with_suffix('.json')
            with open(json_path, 'w') as json_outf:
                json_outf.write( segdict_str )

