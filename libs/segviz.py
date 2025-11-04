"""
Various visualizations for debugging or illustrating segmentation jobs.
"""

import random
from typing import Union,Callable
from pathlib import Path
import json
import logging

import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
from torchvision.tv_tensors import BoundingBoxes, Mask
import skimage as ski
from PIL import Image, ImageDraw

from . import seglib


logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(levelname)s: %(funcName)s - %(message)s", force=True )
logger = logging.getLogger(__name__)


def get_n_color_palette(n: int, s=.85, v=.95) -> list:
    """
    Generate n well-distributed random colors. Use golden ratio to generate colors from the HSV color
    space.

    Reference: https://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically/

    Args:
        n (int): number of color to generate.

    Returns:
        list: a list of (R,G,B) tuples
    """
    golden_ratio_conjugate = 0.618033988749895
    random.seed(13)
    h = random.random()
    palette = np.zeros((1,n,3))
    for i in range(n):
        h += golden_ratio_conjugate
        h %= 1
        palette[0][i]=(h, s, v)
    return (ski.color.hsv2rgb( palette )*255).astype('uint8')[0].tolist()


def batch_label_maps_to_img( inputs:list[Union[Tensor,dict,Path]], raw_maps: list[tuple[np.ndarray,dict]], color_count=-1, alpha=.4):
    """
    Given a list of image tensors and a list of tuples (<labeled map>,<attributes>), returns page images
    with mask overlays, as well as attributes.

    Args:
        inputs (list[Union[Tensor,dict,Path]]): a list of either
            - image tensors (CHW)
         or - dictionaries with 'img' tensor
         or - image paths
        raw_maps (list[tuple[np.ndarray,dict]]): a list of tuples with
            - labeled map (1,H,W)
            - attributes: i.e. dictionary of morphological attributes (simply passed through, for use 
              by a consumer, plotting function) og 
    Returns:
        list[tuple[np.array, dict, str]]: a list of tuples (img_HWC, attributes, id)
    """
    assert (isinstance(inputs[0], Tensor) or ( type(inputs[0]) is dict and 'img' in inputs[0] )) or isinstance(inputs[0], Path)
    
    imgs, ids, maps, attr = [], [], [], []
    if isinstance(inputs[0], Tensor):
        imgs_chw = [ img_chw.cpu().numpy() for img_chw in inputs ] 
        ids = [ f"image-{i}" for i in range(len(imgs_chw)) ]
    elif type(inputs[0]) is dict and 'img' in inputs[0]:
        imgs_chw=[ img['img'].cpu().numpy() for img in inputs ]
        ids = [ img['id'] if 'id' in img else f'image-{i}' for (i,img) in enumerate(inputs) ] 
    elif isinstance(inputs[0], Path):
        imgs_chw,ids=zip(*[ (np.transpose(ski.io.imread(img),(2,0,1)).astype('float32')/255, str(img.name)) for img in inputs ])
    #print([ (Id,img.shape, img.dtype, np.ptp(img)) for img,Id in zip(imgs,ids) ])
    assert all([ img_chw.shape[1:] == mp[0].shape[1:] for img_chw, mp in zip(imgs_chw, raw_maps) ])

    default_color = [0,0,1.0] # BLUE
    for img_chw, mp in zip(imgs_chw, raw_maps):
        # generate labeled masks
        labeled_msk_1hw, attributes = mp
        labeled_msk_hw1 = np.transpose( labeled_msk_1hw, (1,2,0))
        bm_hw1 = labeled_msk_hw1.astype('bool')
        img_hwc = np.transpose( img_chw, (1,2,0))
        img_complementary_hwc = img_hwc * ( ~bm_hw1 + bm_hw1 * (1-alpha))
        col_msk_hwc = None
        if color_count>=0:
            colors = get_n_color_palette( color_count ) if color_count > 0 else get_n_color_palette( np.max(labeled_msk_hw1))
            col_msk_hwc = np.zeros( img_hwc.shape, dtype=img_chw.dtype )
            for l in range(1, np.max(labeled_msk_hw1)+1):
                col = np.array(colors[l % len(colors) ])
                col_msk_hwc += (labeled_msk_hw1==l) * (col/255.0)
            col_msk_hwc *= alpha
        # single color
        else:
            # BLUE * BOOL * ALPHA
            col_msk_hwc = np.full(img_hwc.shape, default_color) * bm_hw1 * alpha
        composed_img_array_hwc = img_complementary_hwc + col_msk_hwc
        # Combination: (H,W,C), i.e. fit for image viewers and plots
        maps.append(composed_img_array_hwc)
        attr.append(attributes)
    
    return list(zip(maps, attr, ids))

def display_segmentation_and_img( img_path: Union[Path,str], segfile: Union[Path,str]=None, segfile_suffix:str='lines.pred.json', show:dict={}, alpha=.4, linewidth=2, out_file='', crop=(1,1), output_file_path='' ):
    """ Render segmentation data on an image.
    The segmentation dictionary is expected to have the following structure:
    
    ```
    { 'regions': [ { 'coords': [[x1,y1], ...,], 'lines': [ {'coords': [[x1,y1], ...,] }, ... }]}
    ```

    Optional, non-standard attributes for the line are handled: 'centerline', 'height'.

    Args:
        img_path (Path): image file
        segfile (Path): if not provided, look for a (XML or JSON) segmentation file that shares its prefix with the image.
        show (dict): features to be shown. Default: `{'polygons': True, 'regions': True, 'baselines': False, 'centerlines': False}`
        alpha (float): overlay transparency.
        linewidth (int): box line width
        output_file_path (str): If path is a directory, save the plot in <output_file_path>/<img_name_stem>.png.; otherwise, save under the provided file path.
        crop (tuple[float,float]): ratio for zoom-in, for x and y respectively.
    """
    
    features = {'polygons': True, 'regions': True, 'baselines': False, 'centerlines': False}
    if show: # because the calling program more likely to pass a list of features to be shown, rather than a dictionary
        features = {'polygons': False, 'regions': False, 'baselines': False, 'centerlines': False}
        features.update( show )

    if segfile is None:
        segfile = str(img_path).replace('.img.jpg', f'.{segfile_suffix}') 
    assert Path(segfile).exists()


    plt.close()
    fig, ax = plt.subplots(figsize=(12,12))

    img_hwc = ski.io.imread( img_path )/255.0
    bm_hw = np.zeros( img_hwc.shape[:2], dtype='bool' )

    segdict = None
    if segfile[-3:]=='xml' or segfile_suffix[-3:]=='xml':
        segdict = seglib.segmentation_dict_from_xml( segfile )
    elif segfile[-4:]=='json' or segfile_suffix[-3:]=='json':
        with open( segfile, 'r' ) as segfile_in:
            segdict = json.load( segfile_in )
    if segdict is None:
        logger.info("Could not parse a valid segmentation dictionary from {}: aborting.".format( segfile ))
        return
        
    if 'image_filename' in segdict and 'image_height' in segdict and 'image_width' in segdict:
        if (img_hwc.shape[0] != segdict['image_height'] or img_hwc.shape[1] != segdict['image_width']):
            logger.info("The size of the provided image ({}) does not match the image properties defined in the segmentation file for {}: aborting.".format(Path(img_path).name, segdict['image_filename']))
            return

    col_msk_hwc = np.zeros( img_hwc.shape, dtype=img_hwc.dtype )
    # for (older) JSON segmentation dictionaries, that have top-level 'lines' list.
    if 'lines' in segdict:
        segdict = seglib.segdict_sink_lines( segdict )
    #regions = [segdict] if 'lines' in segdict else segdict['regions'] 
    for reg in segdict['regions']:
        color_count = len(reg['lines'])
        colors = get_n_color_palette( color_count )
        for l,line in enumerate(reg['lines']):
            col = np.array(colors[l % len(colors) ])
            if features['polygons']:
                rr,cc = (np.array(line['coords']).T)[::-1]
                coords = ski.draw.polygon( rr, cc )
                col_msk_hwc[ coords ] = (col/255.0)
                bm_hw[ coords ] = True
                #plt.plot( cc,rr, linewidth=2 )

            if features['baselines'] and 'baseline' in line:
                baseline_arr = np.array( line['baseline'] )
                plt.plot( *baseline_arr.transpose(), linewidth=1/np.mean(crop))
            if features['centerlines'] and 'centerline' in line:
                centerline_arr = np.array( line['centerline'] )
                plt.plot( *centerline_arr.transpose(), linewidth=1/np.mean(crop))
        
        if features['regions'] and 'coords' in reg:
            reg_closed_boundary = np.array( reg['coords']+[reg['coords'][0]])
            plt.plot( reg_closed_boundary[:,0], reg_closed_boundary[:,1], linewidth=linewidth*1/np.mean(crop))
    col_msk_hwc *= alpha
    bm_hw1 = bm_hw[:,:,None]
    img_complementary_hwc = img_hwc * ( ~bm_hw1 + bm_hw1 * (1-alpha))

    composed_img_array_hwc = img_complementary_hwc + col_msk_hwc

    #with plt.rc_context({'lines.linewidth': .2}):
    plt.imshow( composed_img_array_hwc )
    height, width = img_hwc.shape[:2]
    delta_x, delta_y = (1-crop[0])*width/2, (1-crop[1])*height/2 
    plt.xlim(delta_x, width-delta_x)
    plt.ylim(height-delta_y, delta_y)
    if 'title' in show:
        plt.title( Path(img_path).name )

    if output_file_path:
        output_file_path = Path( output_file_path, img_path.stem).with_suffix('.pdf') if Path(output_file_path).is_dir() else Path(output_file_path)
        plt.savefig( output_file_path, bbox_inches='tight', dpi=fig.dpi )
    else:
        plt.show()


def display_tensor_and_target( img_chw: Tensor, target: dict, alpha=.4, color='g'):
    """ Overlay of instance masks and boxes, no frills: single color.
    Args:
        img_chw (Tensor): (C,H,W) image
        target (dict[str,Tensor]): a dictionary of labels with
        - 'masks'=(N,H,W) tensor of masks, where N=# instances for image
        - 'boxes'=(N,4) tensor of BB coordinates (x1, y1, x2, y2)
        - 'labels'=(N) tensor of box labels
    """
    img_chw = img_chw.detach().numpy()
    mask_nhw = target['masks'].detach().numpy()
    masks_hw = [ m * (m>.5) for m in mask_nhw ]
    boxes = [ [ int(c) for c in box ] for box in target['boxes'].detach().numpy().tolist()]
    bm_hw = np.sum( masks_hw, axis=0).astype('bool')
    col = {'r': [1.0,0,0], 'g':[0,1.0,0], 'b':[0,0,1.0]}[color]
    # RED * BOOL * ALPHA
    red_mask_3hw = np.transpose(np.full(img_chw.shape[1:]+(3,), col),(2,0,1)) * bm_hw * alpha
    img_complementary_chw = img_chw * ( ~bm_hw + bm_hw * (1-alpha))
    composed_img_array_hwc = np.transpose(img_complementary_chw + red_mask_3hw, (1,2,0))
    # x1,y1 ; x2,y1; x2,y2; x1,y2
    polygon_boundaries = [ np.array([[box[0],box[1]], [box[2],box[1]], [box[2],box[3]], [box[0],box[3]], [box[0],box[1]]]) for box in boxes] 
    plt.close()
    plt.imshow( composed_img_array_hwc )
    for i,polyg in enumerate(polygon_boundaries):
        #if i%2 != 0:
        plt.plot(polyg, color=col[color])
    plt.show()


def display_tensor_and_masks( img_chw: Tensor, mask_n1hw: Tensor, alpha=.4, color_count=-1, output_file_path=''):
    """ Overlay of instance binary masks, with choice of colors and option for saving image.

    Args:
        img_chw (Tensor): (C,H,W) image
        mask_n1hw (Tensor): (N,1,H,W) or (N,H,W) tensor of binary masks, where N=# instances for image
        color_count (int): -1 (default) = default color; 0=one color/instance; n>0=n colors
        output_file_path (str): If path is a directory, save the plot in <output_file_path>/<img_name_stem>.png.; otherwise, save under the provided file path.
    """
    img_hwc = img_chw.detach().numpy().transpose(1,2,0)
    if np.max(img_hwc) > 1.0:
        img_hwc = img_hwc.astype('float32')/ 255.0
    mask_n1hw = mask_n1hw.detach().numpy()
    if mask_n1hw.ndim == 3:
        mask_n1hw = mask_n1hw[:,None,:]
    bm_hw1 = np.sum( mask_n1hw, axis=0).astype('bool').transpose(1,2,0)
    col_msk_hwc = None
    default_color = [0,0,1.0] # BLUE
    if color_count>=0:
        colors = get_n_color_palette( color_count ) if color_count > 0 else get_n_color_palette( mask_n1hw.shape[0])
        col_msk_hwc = np.zeros( img_hwc.shape, dtype=img_hwc.dtype )
        for l in range( mask_n1hw.shape[0]):
            col = np.array(colors[l % len(colors) ])
            col_msk_hwc += (mask_n1hw[l].transpose(1,2,0)==1) * (col/255.0)
        col_msk_hwc *= alpha
    else:
        # BLUE * BOOL * ALPHA
        col_msk_hwc = np.full(img_hwc.shape[:2]+(3,), default_color) * bm_hw1 * alpha
    img_complementary_hwc = img_hwc * ( ~bm_hw1 + bm_hw1 * (1-alpha))
    composed_img_array_hwc = img_complementary_hwc + col_msk_hwc
    plt.close()
    fig = plt.figure(figsize=(12,12))
    plt.imshow( composed_img_array_hwc)
    if output_file_path:
        output_file_path = Path( output_file_path, img_path.stem).with_suffix('.png') if Path(output_file_path).is_dir() else Path(output_file_path)
        plt.savefig( output_file_path, bbox_inches='tight' )
    else:
        plt.show()


def display_tensor_and_boxes( img_chw: Tensor, boxes: Tensor, scores: Tensor, threshold=0, alpha=.4, color='g', image_only=False, output_file_path=''):
    """ Overlay of instance boxes with threshold option.
    Args:
        img_chw (Tensor): (C,H,W) image
        boxes (Tensor): (N,4) tensor of BB coordinates (x1, y1, x2, y2)
        scores (Tensor): (N,) tensor of bbox confidence scores.
        threshold (float): bboxes with a score less than <threshold< are drawn in red.
        image_only: show only the image, not the plot axes.
        output_file_path (str): If path is a directory, save the plot in <output_file_path>/<img_name_stem>.png.; otherwise, save under the provided file path.
    """
    img_hwc = img_chw.detach().numpy().transpose(1,2,0)
    boxes = [ [ int(c) for c in box ] for box in boxes.detach().numpy().tolist()]
    polygon_boundaries = [ np.array([[box[0],box[1]], [box[2],box[1]], [box[2],box[3]], [box[0],box[3]],[box[0],box[1]]]) for box in boxes ] 
    plt.close()
    fig = None
    if image_only:
        fig = plt.figure(frameon=False, figsize=(10,10))
        ax = plt.Axes(fig, [0,0,1,1])
        ax.set_axis_off()   
        fig.add_axes(ax)
        ax.set_aspect('equal', adjustable='box')
        ax.imshow( img_hwc, aspect='auto')
    else:
        fig, ax = plt.subplots(figsize=(10,10))
        plt.imshow( img_hwc )
    for i,polyg in enumerate(polygon_boundaries):
        assert scores is not None or threshold==0
        color = 'r' if scores[i]<threshold else 'g'
        plt.plot(polyg[:,0],polyg[:,1], linewidth=3, color=color)
    if output_file_path:
        output_file_path = Path( output_file_path, img_path.stem).with_suffix('.png') if Path(output_file_path).is_dir() else Path(output_file_path)
        plt.savefig( output_file_path, bbox_inches='tight' )
    else:
        plt.show()


def display_soft_masks( masks: Tensor, scores: Tensor, box_threshold=.9, alpha=.4, image_only=False, output_file_path=''):
    """ Display or save soft masks (several plots)
    Args:
        masks (Tensor): (N,H,W) tensor of masks
        threshold (float): box threshold
        output_file_path (str): If path is a directory, save the plots in <output_file_path>/<img_name_stem>-<i>.png.; otherwise, save under the provided file path, with mask/plot number inserted.
    """
    masks = masks.detach().numpy()
    masks = masks[ scores >= box_threshold ]
    plt.close()
    fig, ax = plt.subplots()
    if output_file_path and image_only:
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0,0,1,1])
        ax.set_axis_off()   
        fig.add_axes(ax)

    for i,m in enumerate(masks):
        plt.imshow( m.squeeze())
        if output_file_path:
            output_file_path_rewritten = Path( output_file_path )
            output_file_path_rewritten = output_file_path.joinpath( f"{img_path.stem}-{i}" ).with_suffix('png') if output_file_path_rewritten.is_dir() else Path(f"{output_file_path_rewritten.with_suffix('')}-{i}{output_file_path_rewritten.suffix}")
            plt.savefig( output_file_path_rewritten, bbox_inches='tight' )
        else:
            plt.show()


def img_rgb_to_binary( img_path: Path, alg='otsu' ):
    """
    Binarize an image, with an algorithm of choice.
    """
    color_img = ski.io.imread( img_path )
    img_gray = ski.color.rgb2gray( color_img )
    threshold_func = {
            'otsu': ski.filters.threshold_otsu,
            'niblack': ski.filters.threshold_niblack,
            'sauvola': ski.filters.threshold_sauvola }
    threshold_mask = threshold_func[alg]( img_gray )
    binary_mask = img_gray > threshold_mask
    return ski.util.img_as_ubyte( binary_mask )

def display_mask_heatmaps( masks_chw: Tensor ):
    """ Display heapmap for the combined page masks (sum over boxes).

    Args:
        masks_chw (Tensor): soft masks (C,H,W), with C the number of instances.
    """
    plt.imshow(torch.sum(masks, axis=0).permute(1,2,0).detach().numpy())
    plt.show()
 
def display_line_masks_raw( preds: list[dict], box_threshold=.8, mask_threshold=.2 ):
    """
    For each page, for each box above the threshold, display the line masks in turn.

    Args:
        preds (list[dict]): a list of prediction dictionaries (each with 'masks', 'scores' and 'boxes' entries).
        box_threshold (float): a box whose score does not meet that threshold is ignored.
        mask_threshold (float): a mask pixel whose score does not meet that threshold is ignored.
    """
    for msks,sc in [ (p['masks'].detach().numpy(),p['scores'].detach().numpy()) for p in preds ]:
        for m in msks[sc>box_threshold]:
            m = m[0]
            plt.imshow( m*(m>mask_threshold) )
            plt.show()
