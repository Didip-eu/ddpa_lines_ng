#!/usr/bin/env python3
"""

Line segmentation app: model training, prediction, evaluation.


TODO:

"""

# stdlib
import sys
import json
from pathlib import Path
from typing import Union, Any
import random
import math
import logging
import time
import re

# 3rd party
from PIL import Image
import skimage as ski
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch import nn 
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes, Mask
from torchvision.models.detection import mask_rcnn
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import _default_anchorgen, RPNHead, FastRCNNConvFCHead
from torch.optim.lr_scheduler import ReduceLROnPlateau

# DiDip
import fargv
import tormentor


# local
sys.path.append( str(Path(__file__).parents[1] ))
from libs import seglib
from libs.transforms import build_tormentor_augmentation_for_page_wide_training, build_tormentor_augmentation_for_crop_training, build_tormentor_augmentation_from_list,  ResizeMin
from libs.train_utils import split_set, duration_estimate


logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(levelname)s: %(funcName)s - %(message)s", force=True )
logger = logging.getLogger(__name__)
#logger.propagate=False

p = {
    'max_epoch': 250,
    'max_epoch_force': [-1, "If non-negative, this overrides a 'max_epoch' value read from a resumed model"],
    'img_paths': set(list(Path("dataset").glob('*.img.jpg'))),
    'train_set_limit': [0, "If positive, train on a random sampling of the train set."],
    'validation_set_limit': [0, "If positive, validate on a random sampling of the validation set."],
    'line_segmentation_suffix': ".lines.gt.json",
    'polygon_key': 'boundary',
    'backbone': ('resnet101','resnet50'),
    'lr': 2e-4,
    'img_size': [1024, "Resize the input images to <img_size> * <img_size>; if 'img_height non-null, this determines the width."],
    'img_height': [0, "If non-null, input images are resize to <img_size> * <img_height>"],
    'batch_size': 4,
    'num_workers': 4,
    'patience': 50,
    'param_file': [ 'parameters.json', 'If this file is created _after_ the training has started, it is read at the start of the next epoch (and then deleted), thus allowing to update hyperparameters on-the-fly.'],
    'tensorboard_sample_size': 2,
    'mode': ('train','validate'),
    'weight_file': None,
    'scheduler': 0,
    'scheduler_patience': 15,
    'scheduler_cooldown': 5,
    'scheduler_factor': 0.9,
    'reset_epochs': 0,
    'resume_file': 'last.mlmodel',
    'dry_run': [0, "1: Load dataset and model, but does not actually train: 2: same, but display the validation samples; 3: same as (2) but display also the test samples."],
    'tensorboard': 1,
    'tormentor': 1,
    'device': 'cuda',
    'augmentations': [ set([]), "Pass one or more tormentor class names, to build a choice of training augmentations; by default, apply the hard-coded transformations."],
    'train_style': [('page','patch'), "Use page-wide sample images for training (default), or fixed-size patches."],
    'cache_dir_train': '',
    'cache_dir_val': '',
}

tormentor_dists = {
        'Rotate': tormentor.Uniform((math.radians(-10.0), math.radians(10.0))),
        'Perspective': (tormentor.Uniform((0.85, 1.25)), tormentor.Uniform((.85,1.25))),
        'Wrap': (tormentor.Uniform((0.1, 0.12)), tormentor.Uniform((0.64,0.66))), # no too rough, but intense (large-scale distortion)
        'Zoom': tormentor.Uniform((1.1,1.6)),
        'Brightness': tormentor.Uniform((-0.35,0.35)),
}


class LineDetectionDataset(Dataset):
    """
    PyTorch Dataset for a collection of images and their annotations.
    A sample comprises:

    + image
    + target dictionary: LHW segmentation mask tensor (1 mask for each of the L lines), L4 bounding box tensor, L label tensor.
    """
    def __init__(self, img_paths, label_paths, polygon_key='boundary', transforms=None, img_size=(1024,1024), min_size=0):
        """
        Constructor for the Dataset class.

        Parameters:
            img_paths (list): List of unique identifiers for images.
            label_paths (list): List of label paths.
            polygon_key (str): type of polygon in the segmentation dictionary ('boundary' or 'extBoundary').
            transforms (callable, optional): Optional transform to be applied on a sample.
            img_size (tuple[int]): when default (pre-tormentor) transform resizes the input to a fixed size.
            min_size (int): if non-zero, ensure that image is at least <size_min> on its smaller dimension - used when later augmentations use fixed crops.
        """
        super(Dataset, self).__init__()
        
        self._img_paths = img_paths  # List of image keys
        self._label_paths = label_paths  # List of image annotation files
        self.polygon_key = polygon_key
        self._transforms = transforms if transforms is not None else v2.Compose([
            v2.ToImage(),
            ResizeMin( min_size ) if min_size else v2.Resize( img_size ), 
            v2.SanitizeBoundingBoxes(),
            v2.ToDtype(torch.float32, scale=True),])

        
    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self._img_paths)
        
    def __getitem__(self, index):
        """
        Fetch an item from the dataset at the specified index.

        Args:
            index (int): Index of the item to fetch from the dataset.

        Returns:
            tuple: A tuple containing the image and its associated target (annotations).
        """
        # Retrieve the key for the image at the specified index
        img_path, label_path = self._img_paths[index], self._label_paths[index]
        # Get the annotations for this image
        label_path = Path(str(img_path).replace('.img.jpg','.lines.gt.json'))
        # Load the image and its target (segmentation masks, bounding boxes and labels)
        image, target = self._load_image_and_target(img_path, label_path)
        
        # Apply basic transformations (img -> tensor, resizing, scaling)
        if self._transforms:
            image, target = self._transforms(image, target)

        #print("Image+masks after basic transform:", image.shape, target['masks'].shape)
        return image, target

    def _load_image_and_target(self, img_path, annotation_path):
        """
        Load an image and its target (bounding boxes and labels).

        Parameters:
            img_path (Path): image path
            annotation_path (Path): annotation path

        Returns:
            tuple: A tuple containing the image and a dictionary with 'masks', 'boxes' and 'labels' keys.
        """
        # Open the image file and convert it to RGB
        img = Image.open(img_path)#.convert('RGB')
        #logger.info(img.size)

        with open( annotation_path, 'r') as annotation_if:
            segdict = json.load( annotation_if )
            masks=Mask( seglib.line_binary_mask_stack_from_segmentation_dict(segdict, polygon_key=self.polygon_key))
            labels = torch.tensor( [ 1 ]*masks.shape[0], dtype=torch.int64)
            bboxes = BoundingBoxes(data=torchvision.ops.masks_to_boxes(masks), format='xyxy', canvas_size=img.size[::-1])
            return img, {'masks': masks, 'boxes': bboxes, 'labels': labels, 'path': img_path, 'orig_size': img.size }


    @staticmethod
    def augment_with_bboxes( sample, aug, device ):
        """  Augment a sample (img + masks), and add bounding boxes to the target.
        (For Tormentor only.)

        Args:
            sample (tuple[Tensor,dict]): tuple with image (as tensor) and label dictionary.
        """
        img, target = sample
        img = img.to(device)
        img = aug(img)
        masks, labels = target['masks'].to(device), target['labels'].to(device)
        masks = torch.stack( [ aug(m, is_mask=True) for m in target['masks'] ], axis=0).to(device)

        # first, filter empty masks
        keep = torch.sum( masks, dim=(1,2)) > 10
        masks, labels = masks[keep], labels[keep]
        # construct boxes, filter out invalid ones
        boxes=BoundingBoxes(data=torchvision.ops.masks_to_boxes(masks), format='xyxy', canvas_size=img.shape)
        keep=(boxes[:,0]-boxes[:,2])*(boxes[:,1]-boxes[:,3]) != 0

        target['boxes'], target['labels'], target['masks'] = boxes[keep], labels[keep], masks[keep]
        return (img, target)


class CachedDataset( Dataset ):
    """
    Wrap an existing dataset into a new one, that provides disk-caching functionalities:
    
    + image tensors saved as such
    + annotations saved as Torch pickles
    """

    def __init__(self, data_source:Union[str,Dataset]=None, serialize=False, repeat=1 ):
        self._source_dataset = None
        self._img_paths = []
        self._label_paths = []
        # data source is an existing serialization: load the paths
        if type(data_source) is str:
            assert Path(data_source).is_dir() and Path(data_source).exists()
            self.load( data_source )
        # data source is an existing Dataset: serialize on disk and load the newly generated paths
        elif isinstance(data_source, Dataset) or isinstance(data_source, tormentor.AugmentedDs):
            logger.info("Initialize from {}".format( data_source ))
            assert len(data_source) > 0
            self._source_dataset = data_source
            if serialize:
                self.serialize( repeat=repeat )
            else:
                logger.warning('CachedDataset object has a data source but contains no cached data yet: call serialize() to generate them.')
        else:
            logger.info("Could not identity data_source type")

    def __len__( self ):
        return len( self._img_paths )

    def __getitem__( self, index ):
        """ Load an item from the serialized dataset"""
        assert len(self._img_paths)
        img_chw = torch.load( self._img_paths[index] )
        target = torch.load( self._label_paths[index], weights_only=False )

        #plt.imshow( (img_chw * torch.sum(target['masks'], axis=0)).permute(1,2,0))
        #plt.show()
        return img_chw, target
      
    def _reset( self ):
        self._img_paths = []
        self._label_paths = []

    def serialize( self, subdir='', repeat=1 ):
        # TODO: ensure that reset() empties the cache before
        self._reset()
        root_dir = None
        for r in range(repeat):
            logger.info('serialization: repeat={}'.format(r))
            for (img, target) in tqdm( self._source_dataset ):
                if root_dir is None:
                    dir_path = Path(target['path']).parents[0]
                    root_dir = dir_path.joinpath( subdir )
                    if root_dir != dir_path:
                        if root_dir.exists():
                            for item in [ f for f in root_dir.iterdir() if not f.is_dir()]:
                                item.unlink()
                        else:
                            root_dir.mkdir()
                #plt.imshow( img.permute(1,2,0))
                #plt.show()
                masks = target['masks']
                #plt.imshow( (img * torch.sum(masks, axis=0)).permute(1,2,0))
                #plt.show()
                self._save_sample(img, target, root_dir=root_dir, index=r)

        logger.info('Generated {} samples.'.format( len(self._img_paths)))

    def load( self, cache:str ):
        """
        Load sample image and annotation pathnames from cache directory.
        Image and annotations have same prefix and '.img.plt' and '.plt'
        suffixes, respectively.
        
        Args:
            cache (str): a directory path (created if it does not exist).
        """
        for img_path in Path(cache).glob('*.img.plt'):
            label_path = Path(str(img_path).replace('img.plt', 'plt'))
            assert label_path.exists()
            self._img_paths.append( img_path )
            self._label_paths.append( label_path )
        logger.info('Loaded {} sample names from cache {}.'.format( len(self._img_paths), cache ))
        
    def _save_sample(self, img, target, root_dir='', index=0 ):
        """ Serialize both image and target on disk.
        Args:
            img (Tensor): image
            target (dict): labels (boxes, masks, labels, path)
            root_dir (Path): where to save
            index (int): iteration (when saving multiple flavor of same dataset)
        """
        img_stem = re.sub(r'([^.]+)\..+$', r'\1', target['path'].name )
        stem = f'{img_stem}-{index}'
        # saving image as tensor
        target['path'] = root_dir.joinpath( f'{stem}.img.plt' )
        torch.save( img, target['path'] )
        # saving target as pickle
        target_path = root_dir.joinpath( f'{stem}.plt' )
        torch.save( target, target_path )
        
        self._img_paths.append( target['path'])
        self._label_paths.append( target_path )



def post_process( preds: dict, box_threshold=.9, mask_threshold=.25, orig_size=()):
    """
    Compute lines from predictions, by merging box masks.

    Args:
        preds (dict[str,torch.Tensor]): predicted dictionary for the page:
            - 'scores'(N) : box probs
            - 'masks' (N1HW): line heatmaps
            - 'orig_size': if provided, masks are rescaled to the respective size
    Returns:
         np.ndarray: binary map (1,H,W)
    """
    # select masks with best box scores
    best_masks = [ m.detach().numpy() for m in preds['masks'][preds['scores']>=box_threshold]]
    if len(best_masks) < preds['masks'].shape[0]:
        logger.debug("Selecting masks {} out of {}".format( np.argwhere( preds['scores']>=box_threshold ).tolist(), len(preds['scores'])))
    if not best_masks:
        return None
    # threshold masks
    masks = [ m * (m >= mask_threshold) for m in best_masks ]
    # merge masks 
    page_wide_mask_1hw = np.sum( masks, axis=0 ).astype('bool')
    # optional: scale up masks to the original size of the image
    if orig_size:
        page_wide_mask_1hw = ski.transform.resize( page_wide_mask_1hw, (1, orig_size[1], orig_size[0]))
    return page_wide_mask_1hw


def post_process_boxes( preds: dict, box_threshold=.9, mask_threshold=.1, orig_size=()):
    """
    Compute lines from predictions, by separate processing of box masks.

    Args:
        preds (dict[str,torch.Tensor]): predicted dictionary for the page:
            - 'scores'(N) : box probs
            - 'masks' (N1HW): line heatmaps
            - 'orig_size': if provided, masks are rescaled to the respective size
    Returns:
        tuple[ np.ndarray, list[tuple[int, list, float, list]]]: a pair with
            - binary map(1,H,W)
            - a list of line attribute dicts (label, centroid pt, area, polygon coords, ...)
    """
    # select masks with best box scores
    print("preds['scores'].shape =", preds['scores'].shape)
    print("preds['masks'].shape =", preds['masks'].shape)
    best_masks = [ m.detach().numpy() for m in preds['masks'][preds['scores']>box_threshold]]
    # threshold masks
    masks = [ (m * (m > mask_threshold)).astype('bool') for m in best_masks ]
    # in each mask, keep the largest CC
    clean_masks = []
    for m_1hw in masks:
        print("m_1hw.shape =", m_1hw.shape, "dtype =", m_1hw.dtype)
        labeled_msk_1hw = ski.measure.label( m_1hw, connectivity=2 )
        reg_props = ski.measure.regionprops( labeled_msk_1hw )
        max_label, _ = max([ (reg.label, reg.area) for reg in reg_props ], key=lambda t: t[1])
        clean_masks.append( m_1hw * (labeled_msk_1hw == max_label))

    # merge masks 
    page_wide_mask_1hw = np.sum( clean_masks, axis=0 ).astype('bool')
    plt.imshow( np.squeeze(np.sum( clean_masks, axis=0)) )
    plt.show()
    # optional: scale up masks to the original size of the image
    if orig_size:
        page_wide_mask_1hw = ski.transform.resize( page_wide_mask_1hw, (1, orig_size[1], orig_size[0]))

    return page_wide_mask_1w


def get_morphology( page_wide_mask_1hw: np.ndarray, centerlines=False):
    """
    From a page-wide line mask, extract a labeled map and a dictionary of features.
    
    Args:
        page_wide_mask_1hw (np.ndarray): a binary line mask (1,H,W)
    Returns:
        tuple[ np.ndarray, list[tuple[int, list, float, list]]]: a pair with
            - labeled map(1,H,W)
            - a list of line attribute dicts (label, centroid pt, area, polygon coords, ...)
    """
    
    # label components
    labeled_msk_1hw = ski.measure.label( page_wide_mask_1hw, connectivity=2 )
    logger.debug("Found {} connected components on 1HW binary map.".format( np.max( labeled_msk_1hw )))
    # sort label from top to bottom (using centroids of labeled regions) # note: labels are [1,H,W]. Accordingly, centroids are 3-tuples.
    line_region_properties = ski.measure.regionprops( labeled_msk_1hw )
    logger.debug("Extracted {} region property records from labeled map.".format( len( line_region_properties )))
    # list of line attribute tuples 
    attributes = sorted([ (reg.label, reg.centroid, reg.area ) for reg in line_region_properties ], key=lambda attributes: (attributes[1][1], attributes[1][2]))
    
    max_label = np.max(labeled_msk_1hw[0]).item()
    polygon_coords = []
    line_heights = [-1] * max_label
    skeleton_coords = [ [] for i in range(max_label) ]

    for lbl in range( 1, max_label+1 ):
        lbl_count = np.sum( labeled_msk_1hw[0] == lbl )
        polygon_coords.append( ski.measure.find_contours( labeled_msk_1hw[0] == lbl )[0].astype('int') if lbl_count > 4 else np.array([]))

    if centerlines: 
        page_wide_skeleton_hw = ski.morphology.skeletonize( page_wide_mask_1hw[0] )
        _ , distance = ski.morphology.medial_axis( page_wide_mask_1hw[0], return_distance=True )
        labeled_skl = ski.measure.label( page_wide_skeleton_hw, connectivity=2)
        logger.debug("Computed {} skeletons on 1HW binary map.".format( np.max( labeled_skl )))
        skeleton_coords = [ reg.coords for reg in ski.measure.regionprops( labeled_skl ) ]
        line_heights = []
        logger.debug("Computing line heights...")
        for lbl in range(1, np.max(labeled_skl)+1):
            line_skeleton_dist = page_wide_skeleton_hw * ( labeled_skl == lbl ) * distance 
            logger.debug("- labeled skeleton {} of length {}".format(lbl, np.sum(line_skeleton_dist != 0) ))
            line_heights.append( (np.mean(line_skeleton_dist[ line_skeleton_dist != 0])*2).item() )
        assert len(line_heights) == len( line_region_properties ) 

    # CCs top-to-bottom ordering differ from BBs centroid ordering: usually hints
    # at messy, non-standard line layout
    if [ att[0] for att in attributes ] != list(range(1, np.max(labeled_msk_1hw)+1)):
        logger.warning("Labels do not follow reading order")

    return (labeled_msk_1hw, [{
                'label': att[0], 
                'centroid': att[1], 
                'area': att[2], 
                'polygon_coords': plgc,
                'line_height': lh, 
                'centerline': skc,
            } for att, lh, skc, plgc in zip(attributes, line_heights, skeleton_coords, polygon_coords) ])

def build_nn( backbone='resnet101'):

    if backbone == 'resnet50':
        return maskrcnn_resnet50_fpn_v2(weights=None, num_classes=2)
        
    backbone = resnet_fpn_backbone(backbone_name='resnet101', weights=None)#weights=ResNet101_Weights.DEFAULT)
    rpn_anchor_generator = _default_anchorgen()
    #rpn_anchor_generator = AnchorGenerator(sizes=((128,256,512),),
    #                               aspect_ratios=((1.0, 2.0, 4.0, 8.0),))
    rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], conv_depth=2)
    box_head = FastRCNNConvFCHead(
        (backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d
    )
    return MaskRCNN( 
            backbone=backbone, num_classes=2,
            rpn_anchor_generator=rpn_anchor_generator,
            rpn_head=rpn_head,
            box_head=box_head,)


def predict( imgs: list[Union[str,Path,Image.Image,Tensor,np.ndarray]], live_model=None, model_file='best.mlmodel' ):
    """
    Args:
        imgs (list[Union[Path,Tensor]]): lists of image filenames or tensors; in the latter case, images
            are assumed to have been resized in a previous step (f.i. when predict() is used during the
            training phase and consumes images from the dataset object.
        live_model (SegModel): an instance of the segmentation model class.
        model_file (str): a saved model
    Returns:
        tuple[list[Tensor], list[dict]]: a tuple with 
        - the resized images (as tensors)
        - a list of prediction dictionaries.
        - a list of tuples (width,height) storing size of the original input img
    """
    assert type(imgs) is list

    model = live_model
    if model is None:
        if not Path(model_file).exists():
            return []
        model = SegModel.load( model_file )
    model.net.cpu()
    model.net.eval()

    orig_sizes = []
    img_size = model.hyper_parameters['img_size']
    width, height = (img_size[0],img_size[0]) if len(img_size)==1  else img_size
    tsf = v2.Compose([
        v2.ToImage(),
        v2.Resize([ width,height]),
        v2.ToDtype(torch.float32, scale=True),
    ])
    # every input that is not a tensor needs both resizing and tensor-ification
    if not isinstance(imgs[0], Tensor):
        imgs_live = []
        if isinstance(imgs[0], Path) or type(imgs[0]) is str:
            imgs_live = [ Image.open(img).convert('RGB') for img in imgs ]
        elif isinstance(imgs[0], Image.Image) or type(imgs[0]) is np.ndarray:
            imgs_live = imgs
        imgs, orig_sizes = zip(*[ (tsf(img), (img.size)) for img in imgs_live ])
    else:
        imgs, orig_sizes = [ tsf(img) for img in imgs ], [ img.shape[:0:-1] for img in imgs ]
        
    return (imgs, model.net( imgs ), orig_sizes)
    

class SegModel():

    def __init__(self, backbone='resnet101'):
        self.net = build_nn( backbone )
        self.epochs = []
        self.hyper_parameters = {}

    def save( self, file_name ):
        state_dict = self.net.state_dict()
        state_dict['epochs'] = self.epochs
        state_dict['hyper_parameters']=self.hyper_parameters
        torch.save( state_dict, file_name )

    @staticmethod
    def resume(file_name, reset_epochs=False, **kwargs):
        if Path(file_name).exists():
            state_dict = torch.load(file_name, map_location="cpu")
            epochs = state_dict["epochs"]
            del state_dict["epochs"]
            hyper_parameters = state_dict["hyper_parameters"]
            del state_dict['hyper_parameters']

            model = SegModel( hyper_parameters['backbone'] )
            model.net.load_state_dict( state_dict )

            if not reset_epochs:
                model.epochs = epochs if not reset_epochs else []
                model.hyper_parameters = hyper_parameters
            model.net.train()
            model.net.to( args.device )
            return model
        return SegModel(**kwargs)

    @staticmethod
    def load(file_name, **kwargs):
        if Path(file_name).exists():
            state_dict = torch.load(file_name, map_location="cpu")
            del state_dict["epochs"]
            hyper_parameters = state_dict["hyper_parameters"]
            del state_dict['hyper_parameters']

            model = SegModel( hyper_parameters['backbone'] if 'backbone' in hyper_parameters else 'resnet101')
            model.net.load_state_dict( state_dict )
                                      
            model.hyper_parameters = hyper_parameters
            model.net.eval()
            return model
        logger.warning('Model file {} does not exist: using uinitialized model with default parameters.')
        return SegModel(**kwargs)



# TRAINING AND VALIDATION SCRIPT
if __name__ == '__main__':

    args, _ = fargv.fargv( p )

    hyper_params={ varname:v for varname,v in vars(args).items() if varname in (
        'batch_size', 
        'polygon_key', 
        'backbone',
        'train_set_limit', 
        'validation_set_limit',
        'lr','scheduler','scheduler_patience','scheduler_cooldown','scheduler_factor',
        'max_epoch','patience',
        'augmentations',
        'train_style',
        'tormentor',
        )}
    
    hyper_params['img_size']=[ int(args.img_size), int(args.img_size) ] if not args.img_height else [ int(args.img_size), int(args.img_height) ]

    model = SegModel( args.backbone )
    # loading weights only
    if args.weight_file is not None and Path(args.weight_file).exists():
        logger.info('Loading weights from {}'.format( args.weight_file))
        model.net.load_state_dict( torch.load(args.weight_file, weights_only=True))
    # resuming from dictionary
    elif args.resume_file is not None and Path(args.resume_file).exists():
        logger.info('Loading model parameters from resume file {}'.format(args.resume_file))
        model = SegModel.resume( args.resume_file ) # reload hyper-parameters from there
        hyper_params.update( model.hyper_parameters )
    # TODO: partial overriding of param dictionary 
    # elif args.fine_tune
    if args.max_epoch_force >= hyper_params['max_epoch']:
        hyper_params['max_epoch']=args.max_epoch_force

    model.hyper_parameters = hyper_params

    random.seed(46)
    imgs = list(args.img_paths)
    lbls = [ str(img_path).replace('.img.jpg', args.line_segmentation_suffix) for img_path in imgs ]

    # split sets
    imgs_train, imgs_test, lbls_train, lbls_test = split_set( imgs, lbls )
    imgs_train, imgs_val, lbls_train, lbls_val = split_set( imgs_train, lbls_train )

    if hyper_params['train_set_limit']:
        imgs_train, _, lbls_train, _ = split_set( imgs_train, lbls_train, limit=hyper_params['train_set_limit'])
    if hyper_params['validation_set_limit']:
        imgs_val, _, lbls_val, _ = split_set( imgs_val, lbls_val, limit=hyper_params['validation_set_limit'])

    ds_train, ds_val = None, None
    # Page-wide processing 
    if args.train_style=='page':
        ds_train = LineDetectionDataset( imgs_train, lbls_train, img_size=hyper_params['img_size'], polygon_key=hyper_params['polygon_key'])
        ds_val = LineDetectionDataset( imgs_val, lbls_val, img_size=hyper_params['img_size'], polygon_key=hyper_params['polygon_key'] )

        if args.tormentor:
            aug = build_tormentor_augmentation_for_crop_training( tormentor_dists, crop_size=hyper_params['img_size'][0], crop_before=True )
            ds_train = tormentor.AugmentedDs( ds_train, aug, computation_device=args.device, augment_sample_function=LineDetectionDataset.augment_with_bboxes )
    # Patch-based processing
    elif args.train_style=='patch': # requires Tormentor anyway
        crop_size = hyper_params['img_size'][0]
        if args.cache_dir_train:
            ds_train = CachedDataset( data_source=args.cache_dir_train )
        else:
            # 1. All images resized to at least patch-size
            ds_train = LineDetectionDataset( imgs_train, lbls_train, min_size=crop_size)
            aug = build_tormentor_augmentation_for_crop_training( tormentor_dists, crop_size=crop_size, crop_before=True )
            ds_train = tormentor.AugmentedDs( ds_train, aug, computation_device=args.device, augment_sample_function=LineDetectionDataset.augment_with_bboxes )
            #ds_train = CachedDataset( ds_train )
            #ds_train.serialize( subdir='train_cached', repeat=4 )
        if args.cache_dir_val:
            ds_val = CachedDataset( data_source=args.cache_dir_val )
        else:
            # 2. Validation set should use patch-size crops, not simply resized images!
            ds_val = LineDetectionDataset( imgs_val, lbls_val, min_size=crop_size)
            augCropCenter = tormentor.RandomCropTo.new_size( crop_size, crop_size )
            augCropLeft = tormentor.RandomCropTo.new_size( crop_size, crop_size ).override_distributions( center_x=tormentor.Uniform((0, .6)))
            augCropRight = tormentor.RandomCropTo.new_size( crop_size, crop_size ).override_distributions( center_x=tormentor.Uniform((.4, 1)))
            aug = ( augCropCenter ^ augCropLeft ^ augCropRight ).override_distributions(choice=tormentor.Categorical(probs=(.33, .34, .33)))
            ds_val = tormentor.AugmentedDs( ds_val, aug, computation_device=args.device, augment_sample_function=LineDetectionDataset.augment_with_bboxes )
            #ds_val = CachedDataset( ds_val )
            #ds_val.serialize( subdir='val_cached', repeat=4)
        
    # not used for the moment
    ds_test = LineDetectionDataset( imgs_test, lbls_test, img_size=hyper_params['img_size'], polygon_key=hyper_params['polygon_key'] )

    dl_train = DataLoader( ds_train, batch_size=hyper_params['batch_size'], shuffle=True, collate_fn = lambda b: tuple(zip(*b)))
    dl_val = DataLoader( ds_val, batch_size=1, collate_fn = lambda b: tuple(zip(*b)))

    # update learning parameters from past epochs
    best_loss, best_epoch, lr = np.inf, -1, hyper_params['lr']
    if model.epochs:
        best_epoch,  best_loss = min([ (i, ep['validation_loss']) for i,ep in enumerate(model.epochs) ], key=lambda t: t[1])
        if 'lr' in model.epochs[-1]:
            lr = model.epochs[-1]['lr']
            logger.info("Read start lR from last stored epoch: {}".format(lr))
    logger.info(f"Best validation loss ({best_loss}) at epoch {best_epoch}")

    optimizer = torch.optim.AdamW( model.net.parameters(), lr=lr )
    scheduler = ReduceLROnPlateau( optimizer, patience=hyper_params['scheduler_patience'], factor=hyper_params['scheduler_factor'], cooldown=hyper_params['scheduler_cooldown'])


    def update_parameters( pfile: str, existing_params: dict):
        """ For updating hyperparameters on the fly."""
        pfile = Path( pfile )
        if not pfile.exists():
            return
        with open(pfile,'r') as pfin:
            pdict = json.load( pfin )
            logger.debug("Updating existing params: {} with {}".format(existing_params, pdict))
            existing_params.update( pdict )
        pfile.unlink( missing_ok=True )
        logger.debug("Updated params: {}".format(existing_params))

        
    def update_tensorboard(writer, epoch, scalar_dict):
        if writer is None:
            return
        writer.add_scalars('Loss', { k:scalar_dict[k] for k in ('Loss/train', 'Loss/val') }, epoch)
        for k,v in scalar_dict.items():
            if k == 'Loss/train' or k == 'Loss/val':
                continue
            writer.add_scalar(k, v, epoch)
        return
#        model.net.eval()
#        net=model.net.cpu()
#        inputs = [ ds_val[i][0].cpu() for i in random.sample( range( len(ds_val)), args.tensorboard_sample_size) ]
#        predictions = net( inputs )
#        # (H,W,C) -> (C,H,W)
#        #writer.add_images('batch[10]', np.transpose( batch_visuals( inputs, net( inputs ), color_count=5), (0,3,1,2)))
#        model.net.cuda()
#        model.net.train()
   

    def validate(dry_run=False):
        validation_losses, loss_box_reg, loss_mask = [], [], []
        batches = iter(dl_val)
        for imgs,targets in (pbar := tqdm( batches )):
            if dry_run > 1 and args.device=='cpu':
                fig, ax = plt.subplots()
                img, target = imgs[0], targets[0]
                logger.debug(img.shape, target['path'])
                plt.imshow( img.permute(1,2,0) * torch.sum( target['masks'], axis=0).to(torch.bool)[:,:,None] )
                plt.show(block=True)
                plt.close()
                continue
            pbar.set_description('Validate')
            imgs = torch.stack(imgs).to( args.device )
            targets = [ { k:t[k].to( args.device ) for k in ('labels', 'boxes', 'masks') } for t in targets ]
            loss_dict = model.net(imgs, targets)
            loss = sum( loss_dict.values()) 
            validation_losses.append( loss.detach())
            loss_box_reg.append( loss_dict['loss_box_reg'].detach())
            loss_mask.append( loss_dict['loss_mask'].detach())
        logger.info( "Loss boxes (reg.): {}".format( torch.stack(loss_box_reg).mean().item()))
        logger.info( "Loss boxes (cls.): {}".format( torch.stack(loss_box_cls).mean().item()))
        logger.info( "Loss masks: {}".format( torch.stack(loss_mask).mean().item()))
        return torch.stack( validation_losses ).mean().item()    

    def train_epoch( epoch: int, dry_run=False ):
        
        epoch_losses = []
        batches = iter(dl_train)
        for imgs,targets in (pbar := tqdm(dl_train)):
            if dry_run > 2 and args.device=='cpu':
                fig, ax = plt.subplots(1,len(imgs))
                for i, img, target in zip(range(len(imgs)),imgs,targets):
                    logger.debug(img.shape, target['path'])
                    ax[i].imshow( img.permute(1,2,0) * torch.sum( target['masks'], axis=0).to(torch.bool)[:,:,None] )
                plt.show(block=True)
                plt.close()
                continue
            pbar.set_description(f'Epoch {epoch}')
            imgs = torch.stack(imgs).to( args.device )
            targets = [ { k:t[k].to( args.device ) for k in ('labels', 'boxes', 'masks') } for t in targets ]
            loss_dict = model.net(imgs, targets)
            loss = sum( loss_dict.values())
            
            epoch_losses.append( loss.detach() )
            loss.backward()
                
            # display gradient
            # plt.imshow( imgs[0].grad.permute(1,2,0) )

            optimizer.step()
            optimizer.zero_grad()

        return None if dry_run else torch.stack( epoch_losses ).mean().item() 

    if args.mode == 'train':
        
        model.net.to( args.device )
        model.net.train()
            
        writer=SummaryWriter() if (args.tensorboard and not args.dry_run) else None
        start_time = time.time()
        logger.debug("args.tensorboard={}, writer={}".format(args.tensorboard, writer ))
        
        Path(args.param_file).unlink(missing_ok=True)

        epoch_start = len( model.epochs )
        if epoch_start > 0:
            logger.info(f"Resuming training at epoch {epoch_start}.")

        for epoch in range( epoch_start, hyper_params['max_epoch'] ):

            update_parameters( args.param_file, hyper_params )
            model.hyper_parameters = hyper_params

            epoch_start_time = time.time()
            mean_training_loss = train_epoch( epoch, dry_run=args.dry_run ) # this is where the action happens
            mean_validation_loss = validate( args.dry_run )

            update_tensorboard(writer, epoch, {'Loss/train': mean_training_loss, 'Loss/val': mean_validation_loss, 'Time': int(time.time()-start_time)})

            if hyper_params['scheduler']:
                scheduler.step( mean_validation_loss )
            model.epochs.append( {
                'training_loss': mean_training_loss, 
                'validation_loss': mean_validation_loss,
                'lr': scheduler.get_last_lr()[0],
                'duration': time.time()-epoch_start_time,
            } )
            torch.save(model.net.state_dict() , 'last.pt')
            model.save('last.mlmodel')

            if mean_validation_loss < best_loss:
                logger.info("Mean validation loss ({}) < best loss ({}): updating best model.".format(mean_validation_loss, best_loss))
                best_loss = mean_validation_loss
                best_epoch = epoch
                torch.save( model.net.state_dict(), 'best.pt')
                model.save( 'best.mlmodel' )
            logger.info('Training loss: {:.4f} (lr={}) - Validation loss: {:.4f} - Best epoch: {} (loss={:.4f}) - Time left: {}'.format(
                mean_training_loss, 
                scheduler.get_last_lr()[0],
                mean_validation_loss, 
                best_epoch,
                best_loss,
                duration_estimate(epoch+1, args.max_epoch, model.epochs[-1]['duration']),
                ))
            if epoch - best_epoch > hyper_params['patience']:
                logger.info("No improvement since epoch {}: early exit.".format(best_epoch))
                break

        if writer is not None:
            writer.flush()
            writer.close()

    # validation + metrics
    elif args.mode == 'validate':
        # 1st pass: mask-rcnn validation, for loss
        mean_validation_loss = validate()
        print('Validation loss: {:.4f}'.format(mean_validation_loss))

        # 2nd pass: metrics on post-processed lines
        # use the same model
        pms = []
        for i,sample in enumerate(list(ds_val)[:args.validation_set_limit] if args.validation_set_limit else ds_val):
            img, target = sample
            logger.debug(f'{i}: computing gt_map...', end='')
            gt_map = seglib.gt_masks_to_labeled_map( target['masks'] )
            logger.debug(f'computing pred_map...', end='')
            imgs, preds, _ = predict( [img], live_model=model ) 
            pred_map = np.squeeze(post_process( preds[0], mask_threshold=.2, box_threshold=.75 )[0]) 
            logger.debug(f'computing pixel_metrics')
            pms.append( seglib.polygon_pixel_metrics_two_flat_maps( pred_map, gt_map ))
        print(seglib.mAP( pms ))

        tps, fps, fns = zip(*[ seglib.polygon_pixel_metrics_to_line_based_scores_icdar_2017( pm )[:3] for pm in pms ])
        print("ICDAR 2017")
        print("F1: {}".format( 2.0 * (sum(tps) / (2*sum(tps)+sum(fps)+sum(fns)))))
        print("Jaccard: {}".format(  (sum(tps) / (sum(tps)+sum(fps)+sum(fns)))))

            

