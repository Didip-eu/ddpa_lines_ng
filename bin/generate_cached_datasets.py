#!/usr/bin/env python3

import tormentor
import math
from pathlib import Path
import sys
import fargv
import random

sys.path.append( str(Path(__file__).parents[1] ))

from libs import segviz, transforms as tsf
from bin import ddp_lineseg as lsg
from libs.train_utils import split_set


p = {
        'img_paths': set(list(Path("dataset").glob('*.img.jpg'))),
}


args, _ = fargv.fargv( p )

tormentor_dists = {
        'Rotate': tormentor.Uniform((math.radians(-10.0), math.radians(10.0))),
        'Perspective': (tormentor.Uniform((0.85, 1.25)), tormentor.Uniform((.85,1.25))),
        'Wrap': (tormentor.Uniform((0.1, 0.12)), tormentor.Uniform((0.64,0.66))), # no too rough, but intense (large-scale distortion)
        'Zoom': tormentor.Uniform((1.1,1.6)),
}


random.seed(46)
imgs = list(args.img_paths)
lbls = [ str(img_path).replace('.img.jpg','lines.gt.json') for img_path in imgs ]


imgs_train, imgs_test, lbls_train, lbls_test = split_set( imgs, lbls )
imgs_train, imgs_val, lbls_train, lbls_val = split_set( imgs_train, lbls_train )


ds_train = lsg.LineDetectionDataset( imgs_train, lbls_train, min_size=1024, polygon_key='boundary')
aug = tsf.build_tormentor_augmentation_for_crop_training( tormentor_dists, crop_size=1024, crop_before=True )
ds_train = tormentor.AugmentedDs( ds_train, aug, computation_device='cpu', augment_sample_function=lsg.LineDetectionDataset.augment_with_bboxes )
ds_train_cached = lsg.CachedDataset( data_source = ds_train )
ds_train_cached.serialize( subdir='cached_train', repeat=4)


ds_val = lsg.LineDetectionDataset( imgs_val, lbls_val, min_size=1024, polygon_key='boundary')
augCropCenter = tormentor.RandomCropTo.new_size( crop_size, crop_size )
augCropLeft = tormentor.RandomCropTo.new_size( crop_size, crop_size ).override_distributions( center_x=tormentor.Uniform((0, .6)))
augCropRight = tormentor.RandomCropTo.new_size( crop_size, crop_size ).override_distributions( center_x=tormentor.Uniform((.4, 1)))
aug = ( augCropCenter ^ augCropLeft ^ augCropRight ).override_distributions(choice=tormentor.Categorical(probs=(.33, .34, .33)))
ds_val = tormentor.AugmentedDs( ds_val, aug, computation_device='cpu', augment_sample_function=lsg.LineDetectionDataset.augment_with_bboxes )
ds_val_cached = lsg.CachedDataset( data_source = ds_val )
ds_val_cached.serialize( subdir='cached_val', repeat=4)

ds_test = lsg.LineDetectionDataset( imgs_test, lbls_test, min_size=1024, polygon_key='boundary')
ds_test = tormentor.AugmentedDs( ds_test, aug, computation_device='cpu', augment_sample_function=lsg.LineDetectionDataset.augment_with_bboxes )
ds_test_cached = lsg.CachedDataset( data_source = ds_test )
ds_test_cached.serialize( subdir='cached_test', repeat=4)

