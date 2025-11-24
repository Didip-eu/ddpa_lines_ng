#!/usr/bin/env python3
"""

Line segmentation app: model training.


TODO:

"""

# stdlib
import sys
from pathlib import Path
from typing import Union, Any
import random
import logging
import time

# 3rd party
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# DiDip
import fargv
import tormentor


# local
sys.path.append( str(Path(__file__).parents[1] ))
from libs import seglib
from libs.segmodel import SegModel
from libs.lineseg_datasets import LineDetectionDataset, CachedDataset
from libs.transforms import build_tormentor_augmentation_for_page_wide_training, build_tormentor_augmentation_for_crop_training, build_tormentor_augmentation_from_list,  ResizeMin
from libs.train_utils import split_set, duration_estimate


logging_format="%(asctime)s - %(levelname)s: %(funcName)s - %(message)s"
logging_levels = {0: logging.ERROR, 1: logging.WARNING, 2: logging.INFO, 3: logging.DEBUG }
logging.basicConfig( level=logging.INFO, format=logging_format, force=True )
logger = logging.getLogger(__name__)

p = {
    'max_epoch': 250,
    'max_epoch_force': [-1, "If non-negative, this overrides a 'max_epoch' value read from a resumed model"],
    'img_paths': set(list(Path("dataset").glob('*.img.jpg'))),
    'train_set_limit': [0, "If positive, train on a random sampling of the train set."],
    'validation_set_limit': [0, "If positive, validate on a random sampling of the validation set."],
    'line_segmentation_suffix': ".lines.gt.json",
    'polygon_key': 'coords',
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
    'scheduler': 1,
    'scheduler_patience': 10,
    'scheduler_cooldown': 5,
    'scheduler_factor': 0.8,
    'reset_epochs': 0,
    'resume_file': 'last.mlmodel',
    'dry_run': [0, "1: Load dataset and model, but do not actually train; 2: same, but also display the validation samples."],
    'tensorboard': 1,
    'tormentor': 1,
    'device': 'cuda',
    'augmentations': [ set([]), "Pass one or more tormentor class names, to build a choice of training augmentations; by default, apply the hard-coded transformations."],
    'train_style': [('page','patch'), "Use page-wide sample images for training (default), or fixed-size patches."],
    'cache_dir': ['', ("Location for sample, serialized as Torch tensors. It should contain two subfolders: 'train' and 'val'.") ],
    'verbosity': [2,"Verbosity levels: 0 (quiet), 1 (WARNING), 2 (INFO-default), 3 (DEBUG)"],
}



if __name__ == '__main__':

    args, _ = fargv.fargv( p )

    if args.verbosity != 2:
        logging.basicConfig( level=logging_levels[args.verbosity], format=logging_format, force=True )

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
        model = SegModel.resume( args.resume_file, device=args.device ) # reload hyper-parameters from there
        hyper_params.update( model.hyper_parameters )
    # TODO: partial overriding of param dictionary 
    # elif args.fine_tune
    if args.max_epoch_force >= hyper_params['max_epoch']:
        hyper_params['max_epoch']=args.max_epoch_force

    model.hyper_parameters = hyper_params

    random.seed(46)

    imgs_train, lbls_train, imgs_val, lbls_val = [], [], [], []
    cache_dir_train, cache_dir_val = None, None

    # all of this is not needed if samples already torch-serialized 
    if not args.cache_dir:
        imgs = list(args.img_paths)
        lbls = [ str(img_path).replace('.img.jpg', args.line_segmentation_suffix) for img_path in imgs ]

        # split sets
        imgs_train, imgs_test, lbls_train, lbls_test = split_set( imgs, lbls )
        imgs_train, imgs_val, lbls_train, lbls_val = split_set( imgs_train, lbls_train )

        if hyper_params['train_set_limit']:
            imgs_train, _, lbls_train, _ = split_set( imgs_train, lbls_train, limit=hyper_params['train_set_limit'])
        if hyper_params['validation_set_limit']:
            imgs_val, _, lbls_val, _ = split_set( imgs_val, lbls_val, limit=hyper_params['validation_set_limit'])
    else:
        if not Path( args.cache_dir ).exists():
            logger.error("Dataset cache dir {} does not exist: exiting.".format( args.cache_dir ))
            sys.exit()
        cache_dir_train, cache_dir_val = Path(args.cache_dir).joinpath('train'), Path(args.cache_dir).joinpath('val')
        cache_dir_train.mkdir( exist_ok=True )
        cache_dir_val.mkdir( exist_ok=True )


    ds_train, ds_val = None, None
    # Page-wide processing 
    if args.train_style=='page':
        ds_train = LineDetectionDataset( imgs_train, lbls_train, img_size=hyper_params['img_size'], polygon_key=hyper_params['polygon_key'])
        ds_val = LineDetectionDataset( imgs_val, lbls_val, img_size=hyper_params['img_size'], polygon_key=hyper_params['polygon_key'] )

        if args.tormentor:
            aug = build_tormentor_augmentation_for_crop_training( crop_size=hyper_params['img_size'][0], crop_before=True )
            ds_train = tormentor.AugmentedDs( ds_train, aug, computation_device=args.device, augment_sample_function=LineDetectionDataset.augment_with_bboxes )
    # Patch-based processing
    elif args.train_style=='patch': # requires Tormentor anyway

        if args.cache_dir:

            # This is how you would cache a dataset on-the-fly
            # ds_train = CachedDataset( ds_train )
            # ds_train.serialize( subdir=Path(args.cache_dir).joinpath('train'), repeat=4 )
            # ds_val = CachedDataset( ds_val )
            # ds_val.serialize( subdir=Path(args.cache_dir).joinpath('val'), repeat=4)
            ds_train, ds_val = CachedDataset( data_source=cache_dir_train ), CachedDataset( data_source=cache_dir_val )
            logger.info("Loading disk-cached dataset from {} (train, {} samples) and {} (val, {} samples).".format( cache_dir_train, len(ds_train), cache_dir_val, len(ds_val)))

        else:
            crop_size = hyper_params['img_size'][0]
            # 1. All images resized to at least patch-size
            ds_train = LineDetectionDataset( imgs_train, lbls_train, min_size=crop_size)
            aug = build_tormentor_augmentation_for_crop_training( crop_size=crop_size, crop_before=True )
            ds_train = tormentor.AugmentedDs( ds_train, aug, computation_device=args.device, augment_sample_function=LineDetectionDataset.augment_with_bboxes )
            # 2. Validation set should use patch-size crops, not simply resized images!
            ds_val = LineDetectionDataset( imgs_val, lbls_val, min_size=crop_size)
            augCropCenter = tormentor.RandomCropTo.new_size( crop_size, crop_size )
            augCropLeft = tormentor.RandomCropTo.new_size( crop_size, crop_size ).override_distributions( center_x=tormentor.Uniform((0, .6)))
            augCropRight = tormentor.RandomCropTo.new_size( crop_size, crop_size ).override_distributions( center_x=tormentor.Uniform((.4, 1)))
            aug = ( augCropCenter ^ augCropLeft ^ augCropRight ).override_distributions(choice=tormentor.Categorical(probs=(.33, .34, .33)))
            ds_val = tormentor.AugmentedDs( ds_val, aug, computation_device=args.device, augment_sample_function=LineDetectionDataset.augment_with_bboxes )
        
    if not (len(ds_train) and len(ds_val)):
        logger.warning("Could not load a valid dataset: exiting.")
        sys.exit()

    # not used for the moment
    # ds_test = LineDetectionDataset( imgs_test, lbls_test, img_size=hyper_params['img_size'], polygon_key=hyper_params['polygon_key'] )

    dl_train = DataLoader( ds_train, batch_size=hyper_params['batch_size'], shuffle=True, collate_fn = lambda b: tuple(zip(*b)))
    dl_val = DataLoader( ds_val, batch_size=1, collate_fn = lambda b: tuple(zip(*b)))

    # update learning parameters from past epochs
    best_loss, best_epoch, lr = float('inf'), -1, hyper_params['lr']
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
            logger.info("Updating existing params: {} with {}".format(existing_params, pdict))
            existing_params.update( pdict )
        pfile.unlink( missing_ok=True )
        logger.info("Updated params: {}".format(existing_params))

        
    def update_tensorboard(writer, epoch, scalar_dict):
        if writer is None:
            return
        writer.add_scalars('Loss', { k:scalar_dict[k] for k in ('Loss/train', 'Loss/val') }, epoch)
        for k,v in scalar_dict.items():
            if k == 'Loss/train' or k == 'Loss/val':
                continue
            writer.add_scalar(k, v, epoch)
        return

    def validate(dry_run=False):
        #dict_keys(['loss_classifier', 'loss_box_reg', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg'])
        validation_losses, loss_classifier, loss_box_reg, loss_mask, loss_objectness = [ [] for i in range(5) ]
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
            loss_objectness.append( loss_dict['loss_objectness'].detach())
            loss_classifier.append( loss_dict['loss_classifier'].detach())
            loss_box_reg.append( loss_dict['loss_box_reg'].detach())
            loss_mask.append( loss_dict['loss_mask'].detach())
        logger.info( "Loss objectness: {}".format( torch.stack(loss_objectness).mean().item()))
        logger.info( "Loss classifier: {}".format( torch.stack(loss_classifier).mean().item()))
        logger.info( "Loss boxes (reg.): {}".format( torch.stack(loss_box_reg).mean().item()))
        logger.info( "Loss masks: {}".format( torch.stack(loss_mask).mean().item()))
        return torch.stack( validation_losses ).mean().item()    

    def train_epoch( epoch: int, dry_run=False ):
        
        epoch_losses = []
        batches = iter(dl_train)
        for imgs,targets in (pbar := tqdm(dl_train)):
            if dry_run > 0 and args.device=='cpu':
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


            

