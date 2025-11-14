#!/bin/sh

# update a segmentation metadata file that uses the old format


for i in *.lines.gt.json ; do 
	tr '\n' '@' < $i | perl -pe 's/"image_wh": \[\@\s+(\d+),\@\s+(\d+)\@\s+\],/"image_width": $1,\@    "image_height": $2,/' | tr '@' '\n' > $i.tmp ; 
	mv $i.tmp $i; 
done
