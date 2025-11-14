#!/usr/bin/env python3
"""
From a dataset of images, generate Tormented crops and save them as tensors, for later use in training.

Usage:

```
# generate training and validation sets, with 6 patches out of every source image
PYTHONPATH=. ./bin/ddp_generate_cached_datasets.py -img_paths dataset/*.img.jpg -repeat 6
# generate only validation set
PYTHONPATH=. ./bin/ddp_generate_cached_datasets.py -img_paths dataset/*.img.jpg -repeat 6 -subsets val
```

"""

import tormentor
import math
from pathlib import Path
import sys
import fargv
import random
import torch
import matplotlib.pyplot as plt

sys.path.append( str(Path(__file__).parents[1] ))

from libs import segviz, transforms as tsf
from libs import lineseg_datasets as lsgds
from libs.train_utils import split_set


p = {
        'img_paths': set(list(Path("dataset").glob('*.img.jpg'))),
        'repeat': (1, "Number of patch samples to generate from one image."),
        'img_size': 1024,
        'subsets': set(['train', 'val']),
        'log_tsv': 1,
        'dummy': 0,
        'img_suffix': '.img.jpg',
        'lbl_suffix': '.lines.gt.json',
}


args, _ = fargv.fargv( p )


random.seed(46)
imgs = list([ Path( ip ) for ip in args.img_paths ])
lbls = [ str(img_path).replace(args.img_suffix, args.lbl_suffix) for img_path in imgs ]


imgs_train, imgs_test, lbls_train, lbls_test = split_set( imgs, lbls )
imgs_train, imgs_val, lbls_train, lbls_val = split_set( imgs_train, lbls_train )

if args.log_tsv:
    for subset, log_tsv_file in ((imgs_train, 'train_ds.tsv'), (imgs_val, 'val_ds.tsv'), (imgs_test, 'test_ds.tsv')):
        tsv_path = imgs[0].parent.joinpath(log_tsv_file)
        with open( imgs[0].parent.joinpath(log_tsv_file), 'w') as tsv:
            for path in subset:
                tsv.write('{}\t{}\n'.format(path.name, path.name.replace(args.img_suffix, args.lbl_suffix)))
if args.dummy:
    sys.exit()

# for training, Torment at will
ds_train = lsgds.LineDetectionDataset( imgs_train, lbls_train, min_size=args.img_size, polygon_key='coords')
aug = tsf.build_tormentor_augmentation_for_crop_training( crop_size=args.img_size, crop_before=False )
ds_aug = tormentor.AugmentedDs( ds_train, aug, computation_device='cpu', augment_sample_function=lsgds.LineDetectionDataset.augment_with_bboxes )

for i in range(10):
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(ncols=4, figsize=(15, 4)) 
    ax0.imshow( ds_train[i][0].permute(1,2,0))
    ax1.imshow( torch.sum( ds_train[i][1]['masks'], axis=0) )
    tsf_sample = ds_aug[i]
    ax2.imshow( tsf_sample[0].permute(1,2,0))
    ax3.imshow( torch.sum( tsf_sample[1]['masks'], axis=0) )
    plt.show()

sys.exit()

if 'train' in args.subsets:
    ds_train_cached = lsg.CachedDataset( data_source = ds_train )
    ds_train_cached.serialize( subdir='cached_train', repeat=args.repeat)

# for validation and test, only crops
ds_val = lsgds.LineDetectionDataset( imgs_val, lbls_val, min_size=args.img_size, polygon_key='coords')
augCropCenter = tormentor.RandomCropTo.new_size( args.img_size, args.img_size )
augCropLeft = tormentor.RandomCropTo.new_size( args.img_size, args.img_size ).override_distributions( center_x=tormentor.Uniform((0, .6)))
augCropRight = tormentor.RandomCropTo.new_size( args.img_size, args.img_size ).override_distributions( center_x=tormentor.Uniform((.4, 1)))
aug = ( augCropCenter ^ augCropLeft ^ augCropRight ).override_distributions(choice=tormentor.Categorical(probs=(.33, .34, .33)))
ds_val = tormentor.AugmentedDs( ds_val, aug, computation_device='cpu', augment_sample_function=lsg.LineDetectionDataset.augment_with_bboxes )

if 'val' in args.subsets:
    ds_val_cached = lsg.CachedDataset( data_source = ds_val )
    ds_val_cached.serialize( subdir='cached_val', repeat=args.repeat)

ds_test = lsg.LineDetectionDataset( imgs_test, lbls_test, min_size=args.img_size, polygon_key='coords')
ds_test = tormentor.AugmentedDs( ds_test, aug, computation_device='cpu', augment_sample_function=lsg.LineDetectionDataset.augment_with_bboxes )

if 'test' in args.subsets:
    ds_test_cached = lsg.CachedDataset( data_source = ds_test )
    ds_test_cached.serialize( subdir='cached_test', repeat=4)

