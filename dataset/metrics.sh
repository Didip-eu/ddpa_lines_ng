#!/bin/bash

# Basic image/line metrics

for file in *.lines.gt.json ; do 
	size=$(xli -identify ${file%.lines.gt.json}.img.jpg | perl -pe 's/^.+[^\d](\d+)x(\d+).+$/$1\t$2/') ; 
	line_count=$(grep 'boundary' $file | wc -l) ; 
	let line_count=$line_count-1 ;
	avg_line_height=$(./line_height.py $file)
       	echo -e "$file\t$size\t$line_count\t$avg_line_height" ; 
done
