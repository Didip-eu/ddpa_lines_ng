#!/bin/bash
#
#
# Sanity check on line segmentation dataset
# 
# + check 1-to-1 correspondance between image (JPG) and annotation files (JSON)
# + validate JSON files

USAGE="$0 [<schema_file>]"

if [[ -n "$1" && "$1"=="-h" ]]; then 
	echo $USAGE
	exit 0
fi

FALLBACK_SCHEMA=$(cat<<"EndOfSchema"
{ "id": "https://didip.uni-graz.at/segmentation.schema.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$comment": "Created by NPR on 2026.02.06 - use the following command for validation: check-jsonschema --schemafile schema.json *.lines.gt.json",
  "title": "Page Description",
  "description": "Line segmentation metadata schema, for DiDip/VRE internal use: structure of *.lines.{pred,gt}.json files.",
  "type": "object",
  "required": ["metadata", "image_filename", "image_width", "image_height","regions"],
  "properties": {
    "metadata": {"type": "object","properties": {
       "created": {"type": "string"},
       "creator": {"type": "string"},
       "comment": {"type": "string"}},
       "required": ["created", "creator"]},
    "image_filename": {"type": "string"},
    "image_width": {"type": "integer"},
    "image_height": {"type": "integer"},
    "type": {"type": "string"},
    "text_direction": {"type": "string"},
    "regions": {
      "type": "array",
      "items": {
        "type": "object","required": ["id", "coords"],
        "properties": {
          "id": { "type": "string" },
          "coords": { "type": "array" }, 
          "lines": { 
            "type": "array",
            "items": {
              "type": "object", "required": ["id", "coords", "x-height", "baseline"],
              "properties": {
                "id": { "type": "string" }, 
                "coords": { "type": "array", "items": {"type": "array","items": {"type": "integer"},"minItems": 2,"maxItems": 2}}, 
                "x-height": {"type": "integer"},
                "centerline": { "type": "array","items": {"type": "array","items": {"type": "integer"},"minItems": 2,"maxItems": 2},"minItems": 2},
                "baseline": { "type": "array","items": {"type": "array","items": {"type": "integer"},"minItems": 2,"maxItems": 2},"minItems": 2}}}}}}}}}
EndOfSchema
)


## verify 1-to-1 correspondence between image and GT files, as well as type
for gt in *.gt.json ; do 
	test -f "${gt%.lines.gt.json}.img.jpg" || { 
		echo "Missing file: ${gt%.lines.gt.json}.img.jpg"
		exit 1 
	};
done
for jpg in *.img.jpg ; do 
	test -f "${jpg%.img.jpg}.lines.gt.json" || { 
		echo "Missing file: ${jpg%.img.jpg}.lines.gt.json"
		exit 1 
	};
done

if [[ -n "$1" && -f "$1" ]]; then
	echo "Validate against JSON schema $1..."
	check-jsonschema --schemafile $1 *.lines.gt.json;
else
	echo "No schema file provided for validation: using built-in fallback-schema..."
	echo $FALLBACK_SCHEMA | check-jsonschema --schemafile - *.lines.gt.json;
fi

