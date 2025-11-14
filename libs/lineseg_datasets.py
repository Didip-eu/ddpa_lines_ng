import sys
import logging
from tqdm.auto import tqdm
from pathlib import Path
import json
from typing import Union

import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes, Mask
from PIL import Image


# DiDip
import tormentor

# local
sys.path.append( str(Path(__file__).parents[1] ))
from libs.transforms import ResizeMin
from libs import seglib


logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(levelname)s: %(funcName)s - %(message)s", force=True )
logger = logging.getLogger(__name__)


class LineDetectionDataset(Dataset):
    """
    PyTorch Dataset for a collection of images and their annotations.
    A sample comprises:

    + image
    + target dictionary: LHW segmentation mask tensor (1 mask for each of the L lines), L4 bounding box tensor, L label tensor.
    """
    def __init__(self, img_paths, label_paths, polygon_key='coords', transforms=None, img_size=(1024,1024), min_size=0):
        """
        Constructor for the Dataset class.

        Parameters:
            img_paths (list): List of unique identifiers for images.
            label_paths (list): List of label paths.
            polygon_key (str): type of polygon in the segmentation dictionary ('coords' or 'ext_coords').
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
            tuple[Tensor,dict]: A tuple containing the image (as a tensor) and its associated target (annotations).
        """
        # Retrieve the key for the image at the specified index
        img_path, label_path = self._img_paths[index], self._label_paths[index]
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
            tuple[Image,dict]: A tuple containing the image and a dictionary with 'masks', 'boxes' and 'labels' keys.
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

    def __init__(self, data_source:Union[str,Path,Dataset]=None, serialize=False, repeat=1 ):
        self._source_dataset = None
        self._img_paths = []
        self._label_paths = []
        # data source is an existing serialization: load the paths
        if type(data_source) is str or isinstance(data_source, Path) :
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
                masks = target['masks']
                #plt.imshow( (img * torch.sum(masks, axis=0)).permute(1,2,0))
                #plt.show()
                self._save_sample(img, target, root_dir=root_dir, index=r)

        logger.info('Generated {} samples.'.format( len(self._img_paths)))

    def load( self, cache:str ):
        """
        Load sample image and annotation pathnames from cache directory.
        Image and annotations have same prefix and '.img.pt' and '.lbl.pt'
        suffixes, respectively.
        
        Args:
            cache (str): a directory path (created if it does not exist).
        """
        for img_path in Path(cache).glob('*.img.pt'):
            label_path = Path(str(img_path).replace('img.pt', 'lbl.pt'))
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
        target['path'] = root_dir.joinpath( f'{stem}.img.pt' )
        torch.save( img, target['path'] )
        # saving target as pickle
        target_path = root_dir.joinpath( f'{stem}.lbl.pt' )
        torch.save( target, target_path )
        
        self._img_paths.append( target['path'])
        self._label_paths.append( target_path )

