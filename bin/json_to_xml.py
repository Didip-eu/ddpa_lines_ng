#!/usr/bin/env python3
"""
A stand-alone script for JSON -> PageXML conversion.

The original function is in ddpa_lines_ng/libs/seglib.
"""

import sys
import json
import fargv
from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Union, Any



p = {
    'file_paths': set([]),
    'polygon_key': 'coords',
    'output_format': ('xml', 'stdout'),
    'with_transcription': [1, "Extract line transcription, if it exists"],
    "comment": ['',"A text string to be added to the <Comments> elt."],
}

def xml_from_segmentation_dict(seg_dict: str, pagexml_filename: str='', polygon_key='coords', with_text=True):
    """Serialize a JSON dictionary describing the lines into a PageXML file.
    Caution: this is a crude function, with no regard for validation.

    Args:
         seg_dict (dict[str,Union[str,list[Any]]]): segmentation dictionary of the form

            {"text_direction": ..., "type": "baselines", "lines": [{"tags": ..., "baseline": [ ... ]}]}
            or
            {"text_direction": ..., "type": "baselines", "regions": [ {"id": "r0", "lines": [{"tags": ..., "baseline": [ ... ]}]}, ... ]}
        pagexml_filename (str): if provided, output is saved in a PageXML file (standard output is the default).
        polygon_key (str): if the segmentation dictionary contain alternative polygons (f.i. 'ext_coords'),
            use them, instead of the usual line 'coords'.
    """
    def boundary_to_point_string( list_of_pts ):
        return ' '.join([ f"{pair[0]:.0f},{pair[1]:.0f}" for pair in list_of_pts ] )

    rootElt = ET.Element('PcGts', attrib={
        "xmlns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15", 
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance", 
        "xsi:schemaLocation": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd"})
    metadataElt = ET.SubElement(rootElt, 'MetaData')
    creatorElt = ET.SubElement( metadataElt, 'Creator')
    creatorElt.text=seg_dict['metadata']['creator'] if ('metadata' in seg_dict and 'creator' in seg_dict['metadata']) else 'prov=Universität Graz/DH/nprenet@uni-graz.at'
    createdElt = ET.SubElement( metadataElt, 'Created')
    createdElt.text=datetime.now().isoformat()
    lastChangeElt = ET.SubElement( metadataElt, 'LastChange')
    lastChangeElt.text=createdElt.text
    commentElt = ET.SubElement( metadataElt, 'Comments')
    if 'comments' in seg_dict:
        commentElt.text = seg_dict['comments']
    elif args.comment:
        commentElt.text = args.comment

    img_name = Path(seg_dict['image_filename']).name
    img_width, img_height = seg_dict['image_width'], seg_dict['image_height']    
    pageElt = ET.SubElement(rootElt, 'Page', attrib={'imageFilename': img_name, 'imageWidth': f"{img_width}", 'imageHeight': f"{img_height}"})
    # if no region in segmentation dict, create one (image-wide)
    if 'regions' not in seg_dict:
        seg_dict['regions']=[{'id': 'r0', 'coords': [[0,0],[img_width-1,0],[img_width-1,img_height-1],[0,img_height-1]]}, ]
    for reg in seg_dict['regions']:
        reg_xml_id = f"r{reg['id']}" if type(reg['id']) is int else f"{reg['id']}"
        regElt = ET.SubElement( pageElt, 'TextRegion', attrib={'id': reg_xml_id})
        ET.SubElement(regElt, 'Coords', attrib={'points': boundary_to_point_string(reg['coords'])})
        # 3 cases: 
        # - top-level list of lines with region ref
        # - top-level list of lines with no regions
        # - top-level regions with a list of lines in each
        lines = [ l for l in seg_dict['lines'] if (('region' in l and l['region']==reg['id']) or 'region' not in l) ] if 'lines' in seg_dict else reg['lines']
        for line in lines:
            textLineElt = ET.SubElement( regElt, 'TextLine', attrib={'id': f"{reg_xml_id}l{line['id']}" if type(line['id']) is int else f"{reg['id']}{line['id']}"} )
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



if __name__ == '__main__':

    args, _ = fargv.fargv( p )

    for json_path in args.file_paths:
        xml_path = json_path.replace('.lines.gt.json', '.xml')
        json_path = Path( json_path )

        with open( json_path, 'r') as json_if:
            segdict = json.load( json_if )
            if args.output_format == 'stdout':
                xml_from_segmentation_dict( segdict, '', polygon_key=args.polygon_key, with_text=args.with_transcription )
            else:
                xml_from_segmentation_dict( segdict, xml_path, polygon_key=args.polygon_key, with_text=args.with_transcription )

