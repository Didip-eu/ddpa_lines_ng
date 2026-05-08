#!/usr/bin/env python3
"""
PageXML -> JSON conversion.
"""

import sys
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Union, Any

import fargv
from fargv import FargvChoice, FargvInt, FargvFloat, FargvPositional, FargvTuple
from jsonschema import validate

src_root = Path(__file__).parents[1]
sys.path.append( str( src_root ))
from libs import seglib, segformats as sgf



p = {
    'file_paths': FargvPositional(default=[]),
    'output_format': FargvChoice(['json', 'stdout'], description="Output format"),
    'input_suffix': '.xml',
    'get_text': (True, "Extract text content of the line, if it exists"),
    'overwrite_existing': (False, "Overwrite an existing file."),
    "comment": ('',"A text string to be added to the <Comments> elt."),
    "verbose": False,
    "validate": (False, "Validate against a JSON schema."),
    "json_schema": ('', "JSON schema file to use: if empty, the built-in schema is used.")
}


schema_dict = { 
  "id": "https://didip.uni-graz.at/segmentation.schema.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$comment": "Created by NPR on 2026.02.06 - use the following: 'check-jsonschema --schemafile schema.json *.lines.gt.json' for CLI validation or 'jsonschema.validate(instance=dict, schema=dict)' for in-script validation.",
  "title": "Page Description",
  "description": "Line segmentation metadata schema, for DiDip/VRE internal use: structure of *.lines.{pred,gt}.json files.",
  "type": "object",
  "required": ["metadata", "image_filename", "image_width", "image_height","regions"],
  "properties": {
    "metadata": {
      "type": "object",
      "properties": {
        "created": { "type": "string" },
        "creator": { "type": "string" },
        "comment": { "type": "string" } },
      "required": ["created", "creator"] },
    "image_filename": { "type": "string" },
    "image_width": { "type": "integer" },
    "image_height": { "type": "integer" },
    "type": { "type": "string" },
    "text_direction": { "type": "string" },
    "lines": {"not":{}},
    "regions": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "coords"],
        "properties": {
          "id": { "type": "string" },
          "coords": { "type": "array" }, 
          "lines": { 
            "type": "array",
            "items": {
              "type": "object", 
              "required": ["id", "coords", "baseline"],
              "properties": { 
                "id": { "type": "string" }, 
                "coords": { 
                  "type": "array",
                  "items": {
                    "type": "array",
                    "items": { "type": "integer" },
		    "minItems": 2,
		    "maxItems": 2 } }, 
                "x-height": { "type": "integer" },
                "centerline": { 
                  "type": "array",
                  "items": {
                    "type": "array",
                    "items": { "type": "integer" },
		    "minItems": 2,
		    "maxItems": 2 },
		  "minItems": 2 },
                "baseline": { 
                  "type": "array",
                  "items": {
                    "type": "array",
                    "items": { "type": "integer" },
		    "minItems": 2,
		    "maxItems": 2 },
		  "minItems": 2 } } } } } } } } }

if __name__ == '__main__':

    args, _ = fargv.parse( p )

    if args.validate:
        if args.json_schema and Path(args.json_schema).exists():
            with open( args.json_schema ) as sch_if:
                schema_dict = json.load( sch_if )
                if args.verbose:
                    print(f"Using schema file {args.json_schema} for validation.")
        elif args.verbose:
            print("Using built-in JSON schema for validation.")

    for xml_path in args.file_paths:

        xml_path = Path(xml_path)
        if args.verbose:
            print(xml_path)

        segdict = sgf.segmentation_dict_from_xml( xml_path, get_text=args.get_text )
        segdict = sgf.segdict_sink_lines( segdict )

        # Raise an exception if invalid
        if args.validate:
            validate( instance=segdict, schema=schema_dict )

        segdict_str = json.dumps( segdict, indent=2 )
        if args.output_format == 'stdout':
            print( segdict_str )
        else:
            json_path = Path(str(xml_path).replace(args.input_suffix, '.json'))
            if not args.overwrite_existing and json_path.exists():
                print("File {} exists: abort.".format( json_path ))
            elif not re.search( r'{}$'.format(args.input_suffix), xml_path.name):
                print(f"Input file path '{xml_path.name}' does not match input suffix '{args.input_suffix}': output aborted.")
            else:
                with open(json_path, 'w') as json_outf:
                    json_outf.write( segdict_str )

