#!/usr/bin/env python3
"""
Basic search functionalities for FSDB (modified from Ch-Lat).

This module is _not_ intended for production: it supports auxiliary, internal tasks
where quick matching of images to charters is needed, such as presentations, 
plotting scripts etc.
"""

from pathlib import Path
import re
import json
import itertools
import base64
from datetime import datetime
import sys 
import io
import numpy as np

# third-party
import jsonschema
from PIL import Image, ImagePath, ImageDraw


def lemmatize( p:Path, suffix='', replacement=''):
    if suffix:
        return re.sub(r'(.+).{}$'.format(suffix), r'\1.{}'.format( replacement ) if replacement else r'\1', str(p))
    return re.sub(r'\..+', f'.{replacement}' if replacement else '' ,  str(p))


class Fsdb:

    def __init__(self, fsdb_root, conf={} ):
        if not Path(fsdb_root).exists():
            raise FileNotFoundError()
        self.fsdb_root = fsdb_root
        self.config = {
                'flat': False,
                'charter_img_suffix': 'img.jpg',
                'gt_htr_suffix': 'htr.gt.json',
                'pred_seg_suffix': 'lines.pred.json',
                'pred_htr_suffix': 'htr.pred.json',
        }
        self.config.update( conf )

    class Charter:
        
        url_filename = 'CH.url.txt'
        atom_id_filename = 'CH.atom_id.txt'
        cei_filename = 'CH.cei.xml'

        def __init__(self, charter_path, fsdb_root ):
            charter_path = Path( charter_path )
            if not charter_path.exists():
                raise FileNotFoundError()
            self.path = str(charter_path)
            self.fsdb_path = str(charter_path).replace( str(fsdb_root), '')
            self.md5 = charter_path.name
            self.url = open( charter_path.joinpath( self.url_filename )).read()
            self.atom_id = open( charter_path.joinpath( self.atom_id_filename )).read()
            self.cei = open( charter_path.joinpath( self.cei_filename )).read()

        def to_json(self)->str:
            return { 
                    "path": self.path,
                    "fsdb_path": self.fsdb_path,
                    "md5": self.md5,
                    "url": self.url,
                    "atom_id": self.atom_id }

        def __str__(self):
            return json.dumps( self.to_json())


    def stats(self, ):
        """ A few stats about this database."""
        
        img_sfx, lines_gt_sfx, lines_pred_sfx, htr_pregt_sfx, htr_gt_sfx = [ self.config[k] for k in ('charter_img_suffix', 
                                    'gt_seg_suffix', 'pred_seg_suffix', 
                                    'pregt_htr_suffix', 'gt_htr_suffix') ]
        all_charters = list(Path(self.fsdb_root).glob('*/*/*/CH.cei.xml')) if not self.config['flat'] else list(Path(self.fsdb_root).glob('*.{}'.format(img_sfx)))
        
        # flat folder (archive_id ignored)
        if self.config['flat']:
            charter_img_paths=list(Path(self.fsdb_root).glob('*.{}'.format(img_sfx)))
        elif self.config['crop']:
            charter_img_paths=list(Path(self.fsdb_root).glob('*/*/*/*.seals.crops/*.{}'.format(img_sfx)))
        else:
            charter_img_paths=list(Path(self.fsdb_root).glob('*/*/*/*.{}'.format(img_sfx)))

        current_state = [ 
                ( Path(lemmatize(p, suffix=img_sfx, replacement=lines_pred_sfx )).exists(), Path(lemmatize(p, suffix=img_sfx, replacement=lines_gt_sfx )).exists(),
                  Path(lemmatize(p, suffix=img_sfx, replacement=htr_pregt_sfx )).exists(), Path(lemmatize(p, suffix=img_sfx, replacement=htr_gt_sfx )).exists()) for p in charter_img_paths ]
        report = { "total_charters": len(all_charters), "total_images": len(current_state), 
                "lines_pred_count": len(list(itertools.filterfalse(lambda t: not t[0], current_state))), "lines_gt_count": len(list(itertools.filterfalse(lambda t: not t[1], current_state))),
                "htr_pregt_count": len(list(itertools.filterfalse(lambda t: not t[2], current_state))), "htr_gt_count": len(list(itertools.filterfalse(lambda t: not t[3], current_state)))
                }
        report['lines_pred_ratio']=round(float(report['lines_pred_count']/report['total_images']),2)
        report['lines_gt_ratio']=round(float(report['lines_gt_count']/report['total_images']),2)
        report['htr_pregt_ratio']=round(float(report['htr_pregt_count']/report['total_images']),2)
        report['htr_gt_ratio']=round(float(report['htr_gt_count']/report['total_images']),2)

        return report



    def read_segmentation_file(self, archive_id:str, charter_img_id: str, suffix: str ) -> dict:
        """
        Args:
            archive_id (str): archive name
            charter_img_id (str): charter atom id
            suffix (str): 'lines.gt.json' (GT) or 'lines.pred.json' (prediction).
        Returns:
            dict: a segmentation dictionary.
        """
        return self.read_img_metadata(archive_id, charter_img_id, suffix)

    def get_archives(self, ) -> list[str]:
        """
        Get a list of all archive directories.

        Returns:
            list[str]: a list of archive names.
        """
        archives = list([ p.name for p in Path( self.fsdb_root).glob('*') if p.is_dir() and re.match(r'[A-Z]{2}-[A-Za-z]+', p.name)])
        archives.append('COLLECTIONS')
        return sorted(archives)


    def get_charter_images(self, archive_id:str='') -> tuple[str,dict]:
        """
        For given archive, get a map of all images, with their attributes.

        Args:
            archive_id (str): the name of an archive directory; if empty, the first archive in the list is used.
        Returns:
            tuple[str,dict]: a pair with the archive id passed to the function, as well as a dictionary
                with image ids as keys and a dictionary of image attributes (filename, segmentation data, ...)
                as value. 
        """
        if not archive_id:
            if self.config['flat']:
                archive_id='stand-alone'
            else:
                archive_id = sorted([ p.name for p in Path( self.fsdb_root).glob('*') if p.is_dir() and p.name != '.git'])[0]
        charter_images = []
        #print(list(Path(self.fsdb_root).glob('*.{}'.format( self.config['charter_img_suffix']))))
        # Flat file system, crop-like by default
        if self.config['flat']:
            charter_images = [ {'id': lemmatize(img.name, suffix=self.config['charter_img_suffix']), 'archive': archive_id, 'filename': str(img), 'gtsegfile': None} for img in sorted(Path(self.fsdb_root).glob('*.{}'.format( self.config['charter_img_suffix']))) ]
        # FSDB file system, crops only
        elif self.config['crop']:
            charter_images = [ {'id': lemmatize(img.name, suffix=self.config['charter_img_suffix']), 'archive': archive_id, 'filename': str(img), 'gtsegfile': None} for img in sorted(Path(self.fsdb_root).glob('{}/*/*/*.seals.crops/*.{}'.format( archive_id, self.config['charter_img_suffix']))) ]
        # FSFB file system, full-page
        else:
            charter_images = [ {'id': lemmatize(img.name, suffix=self.config['charter_img_suffix']), 'archive': archive_id, 'filename': str(img), 'gtsegfile': None} for img in sorted(Path(self.fsdb_root).glob('{}/*/*/*.{}'.format( archive_id, self.config['charter_img_suffix']))) ]

        for number, ch_img in enumerate(charter_images, start=1):
            filepath_stem = lemmatize( Path(ch_img['filename']), suffix=self.config['charter_img_suffix'] )
            ch_img['number']=number
            if self.config['crop']:
                ch_img['charter']=Path(ch_img['filename']).parent.parent.name
            else:
                ch_img['charter']=Path(ch_img['filename']).parent.name

            gt_seg_filename = '{}.{}'.format( filepath_stem, self.config['gt_seg_suffix'])
            if Path( gt_seg_filename ).exists():
                ch_img['hasGTData']=True
            pred_seg_filename = '{}.{}'.format( filepath_stem, self.config['pred_seg_suffix'] )
            if Path( pred_seg_filename ).exists():
                ch_img['hasPredData']=True
        
        return archive_id, charter_images


    def get_image(self, archive_id: str, charter_img_id:str):
        """
        Find an image from the given archive, given its id.

        Args:
            archive_id (str): the name of an archive directory.
            charter_img_id (str): the image id.
        Returns:
            bytes: an array of bytes.
        """
        charter_img_path = self.search( archive_id, charter_img_id )
        if charter_img_path is None:
            return None
        try:
            data = open( charter_img_path[0], 'rb' ).read()
            return data
        except (IOError) as e:
            print("Could not open {}".format( charter_img_path[0]))
            return None


    def read_lines(self, charter_img_id:str, data_type='pregt', polygon_key='coords',  polygon_mask=True):
        """ Read line items, and return line crops with their transcriptions.
        Output:
            tuple[list[list],int}]: a list of line descriptors, as well as the maximum
                width of a line (character length).
        """

        htr_suffix = self.config['gt_htr_suffix'] if data_type=='gt' else self.config['pregt_htr_suffix']
        charter_img_path = self.search( '*', charter_img_id )
        charter_htr_path = self.search( '*', charter_img_id, suffix=htr_suffix)
        issues_path = self.search('*', charter_img_id, suffix=self.config['issues_htr_suffix'])

        if charter_img_path is None or charter_htr_path is None:
            return ([], -1)
        else:
            charter_img_path = charter_img_path[0]
            charter_htr_path = charter_htr_path[0]
        try:
            page_img = Image.open( charter_img_path, 'r')
            page_dict = json.load( open(charter_htr_path, 'r') )
            issues = json.load( open(issues_path[0], 'r')) if issues_path else []
            line_tuples = []
            max_width = 0
            lines = itertools.chain.from_iterable( [ reg['lines'] for reg in page_dict['regions'] if 'lines' in reg ] )
            have_issue = { issue['line_id']:True for issue in issues }
            
            for tl in lines:
                baseline_coordinates = [ tuple(pair) for pair in tl['baseline'] ]
                baseline_array = np.array( tl['baseline'])
                baseline_min, baseline_max = np.min( baseline_array, axis=0), np.max( baseline_array, axis=0)
                assert baseline_min.shape == (2,)
               
                polygon_coordinates = [ tuple(pair) for pair in tl[polygon_key]]
                polygon_array = np.array( tl[polygon_key] )
                polygon_min, polygon_max = np.min( polygon_array, axis=0 ), np.max( polygon_array, axis=0 )
                assert polygon_min.shape==(2,)

                # extend the boundaries to the baseline's or polygon's bbox dimensions, whichever is the larger
                offset_xy, extent_xy = np.minimum(baseline_min, polygon_min), np.maximum(baseline_max, polygon_max)
                polygon_transposed = polygon_array - offset_xy # for mask 
                textline_bbox = ( *offset_xy, *extent_xy )
                bbox_width = textline_bbox[2]-textline_bbox[0]
                if bbox_width > max_width: 
                    max_width = bbox_width
                page_img_crop = page_img.crop( textline_bbox )
                #----- polygon mask
                if polygon_mask:
                    gray_box = Image.new( page_img_crop.mode, page_img_crop.size, 0)
                    polygon_mask = Image.new( 'L', page_img_crop.size, 150 )
                    ImageDraw.Draw( polygon_mask ).polygon( [ tuple(pair) for pair in polygon_transposed.tolist()], fill=255)
                    page_img_crop=Image.composite(page_img_crop, gray_box, polygon_mask)
                imgByteArr = io.BytesIO()
                page_img_crop.save( imgByteArr, format='PNG')
                line_tuples.append( [
                    tl['id'], 
                    tl['text'] if 'text' in tl else '', 
                    base64.b64encode(imgByteArr.getvalue()).decode(), 
                    bbox_width,  
                    tl['id'] in have_issue])

            return (line_tuples, max_width) #json.dumps(line_tuples)
        except (IOError) as e:
            print("Could not open{}".format( charter_img_path))
            return ([], -1)

                
    def get_path(self,  archive_id:str='*', charter_id:str='*', charter_img_id:str='*', img_suffix:str=None) -> list[Path]:
        """
        Search specific image or charter paths.

        Args:
            archive_id (str): archive md5 or wildard.
            charter_id (str): charter md5 or wildard.
            charter_img_id (str): charter_img md5 or wildard.
            img_suffix (str): image suffix (by default: self.config['charter_img_suffix'])

        Returns:
            list[Path]: a list of paths.
        """
        if img_suffix is None:
            img_suffix = self.config['charter_img_suffix']
        if self.config['flat']:
            file_paths=Path(self.fsdb_root).glob('{}.{}'.format(charter_img_id, img_suffix))
        # flat folder (archive_id ignored)
        else:
            search_path = '{}/*/{}/{}.{}'.format(archive_id, charter_id, charter_img_id, img_suffix)
            file_paths=Path(self.fsdb_root).glob(search_path)
        if not file_paths:
            return None
        return list(sorted(file_paths))

    def get_charter(self, archive_id:str='*', charter_id:str='*', charter_img_id:str='*', img_suffix:str=None) -> list[Charter]:
        """
        Returns charter record. Results contain all unique charter paths. When using the 
        image id as a key, an image that has two instances in the FSDB tree will result into 2
        charter entries in the results.

        Args:
            archive_id (str): archive md5 or wildard.
            charter_id (str): charter md5 or wildard.
            charter_img_id (str): charter_img md5 or wildard.
            img_suffix (str): image suffix (by default: self.config['charter_img_suffix'])

        Returns:
            list[Fsdb.Charter]: a list of Charter objects
        """
        if img_suffix is None:
            img_suffix = self.config['charter_img_suffix']
        search_path = '{}/*/{}/{}.{}'.format(archive_id, charter_id, charter_img_id, img_suffix)
        print(search_path)
        img_paths=Path(self.fsdb_root).glob(search_path)
        if not img_paths:
            return None
        charters = {}
        for p in img_paths:
            charter_path = str(p.parent)
            charters[charter_path]=self.Charter( charter_path, self.fsdb_root )
        return list(charters.values())


    def read_img_metadata(self, archive_id:str, charter_img_id:str, suffix=None):
        if suffix is None:
            return {}

        infile, returnValue = None, {}
        #print(f"self.search( {archive_id}, {charter_img_id}, suffix={suffix})")
        data_path = self.search( archive_id, charter_img_id, suffix=suffix)
        #print(f"data_path={data_path}")
        if not data_path:
            return {}
        try:
            infile = open(data_path[0], 'r') 
            returnValue = json.load( infile );
        except (IOError, FileNotFoundError) as e:
            pass
        finally:
            if infile is not None:
                infile.close();
        return returnValue;

############

def test_init():
    fsdb = Fsdb(Path.home().joinpath('tmp/data/fsdb'))
    assert fsdb.fsdb_root == Path.home().joinpath('tmp/data/fsdb')

def test_get_path():
    fsdb = Fsdb(Path.home().joinpath('tmp/data/fsdb'))
    assert len(fsdb.get_path( charter_img_id = '6f3d59fd26cb726b98e64f4d8a200dcc' )) > 0
    
def test_get_charter_by_charter():
    fsdb = Fsdb(Path.home().joinpath('tmp/data/fsdb'))
    charters_found = fsdb.get_charter( charter_id = 'e36ec65a5592bd9c962ccdcdf8304a39' )
    print([ ch.to_json() for ch in charters_found])
    assert len(charters_found) > 0
    assert charters_found[0].url == "https://www.monasterium.net/mom/CH-StiASG/Urkunden/FF.5.J.33/charter"

def test_get_charter_by_img():
    fsdb = Fsdb(Path.home().joinpath('tmp/data/fsdb'))
    charters_found = fsdb.get_charter( charter_img_id = '6f3d59fd26cb726b98e64f4d8a200dcc' )
    assert len(charters_found) > 0
    assert charters_found[0].url == "https://www.monasterium.net/mom/AT-StiAL/LambachOSB/1268/charter"


