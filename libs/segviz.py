"""
Various visualizations for debugging or illustrating segmentation jobs.
"""

import random
from typing import Union,Callable
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
from torchvision.tv_tensors import BoundingBoxes, Mask
import skimage as ski


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


def batch_visuals( inputs:list[Union[Tensor,dict,Path]], raw_maps: list[tuple[np.ndarray,dict]], color_count=-1, alpha=.4):
    """
    Given a list of image tensors and a list of tuples (<labeled map>,<attributes>), returns page images
    with mask overlays, as well as attributes.

    Args:
        inputs (list[Tensor]): a list of 
            - image tensors
            - dictionaries with 'img' tensor
            - image paths
        raw_maps (list[tuple[np.ndarray,dict]]): a list of tuples with
            - labeled map (1,H,W)
            - attributes: i.e. dictionary of morphological attributes (simply passed through, for use 
              by a consumer, plotting function)
    Returns:
        list[tuple[np.array, dict, str]]: a list of tuples (img_HWC, attributes, id)
    """
    assert (isinstance(inputs[0], Tensor) or ( type(inputs[0]) is dict and 'img' in inputs[0] )) or isinstance(inputs[0], Path)
    
    imgs, ids, maps, attr = [], [], [], []
    if isinstance(inputs[0], Tensor):
        imgs = [ img.cpu().numpy() for img in inputs ] 
        ids = [ f"image-{i}" for i in range(len(imgs)) ]
    elif type(inputs[0]) is dict and 'img' in inputs[0]:
        imgs=[ img['img'].cpu().numpy() for img in inputs ]
        ids = [ img['id'] if 'id' in img else f'image-{i}' for (i,img) in enumerate(inputs) ] 
    elif isinstance(inputs[0], Path):
        imgs,ids=zip(*[ (np.transpose(ski.io.imread(img),(2,0,1)).astype('float32')/255, str(img.name)) for img in inputs ])
    #print([ (Id,img.shape, img.dtype, np.ptp(img)) for img,Id in zip(imgs,ids) ])
    assert all([ img.shape[1:] == mp[0].shape[1:] for img,mp in zip(imgs,raw_maps) ])

    default_color = [0,0,1.0] # BLUE
    for img,mp in zip(imgs,raw_maps):
        # generate labeled masks
        labeled_msk, attributes = mp
        labeled_msk = np.transpose( labeled_msk, (1,2,0))
        bm = labeled_msk.astype('bool')
        img = np.transpose( img, (1,2,0))
        img_complementary = img * ( ~bm + bm * (1-alpha))
        col_msk = None
        if color_count>=0:
            colors = get_n_color_palette( color_count ) if color_count > 0 else get_n_color_palette( np.max(labeled_msk))
            col_msk = np.zeros( img.shape, dtype=img.dtype )
            for l in range(1, np.max(labeled_msk)+1):
                col = np.array(colors[l % len(colors) ])
                col_msk += (labeled_msk==l) * (col/255.0)
            col_msk *= alpha
        # single color
        else:
            # BLUE * BOOL * ALPHA
            col_msk = np.full(img.shape, default_color) * bm * alpha
        composed_img_array = img_complementary + col_msk
        # Combination: (H,W,C), i.e. fit for image viewers and plots
        maps.append(composed_img_array)
        attr.append(attributes)
    
    return list(zip(maps, attr, ids))

def display_segmentation_and_img( img_path: Union[Path,str], segfile: Union[Path,str]=None, segfile_suffix:str='lines.pred.json', show:dict={}, alpha=.4, linewidth=2 ):
    """ Render segmentation data on an image.
    The segmentation dictionary is expected to have the following structure:
    
    ```
    { 'regions': [ { 'boundary': [[x1,y1], ...,], 'lines': [ {'boundary': [[x1,y1], ...,] }, ... }]}
    ```

    Args:
        img_path (Path): image file
        segfile (Path): if not provided, look for a segmentation file that shares its prefix with the image.
        show (dict): features to be shown. Default: `{'polygons': True, 'regions': True, 'baselines': False}`
        linewidth (int): box line width
    """
    
    features = {'polygons': True, 'regions': True, 'baselines': False}
    if show: # because the calling program more likely to pass a list of features to be shown, rather than a dictionary
        features = {'polygons': False, 'regions': False, 'baselines': False}
        features.update( show )

    if segfile is None:
        segfile = str(img_path).replace('.img.jpg', f'.{segfile_suffix}') 
    assert Path(segfile).exists()

    plt.close()
    fig, ax = plt.subplots()

    img_hwc = ski.io.imread( img_path )/255.0
    bm_hw = np.zeros( img_hwc.shape[:2], dtype='bool' )
    with open( segfile, 'r' ) as segfile_in:
        segdict = json.load( segfile_in )
        
        if 'imageFilename' in segdict and 'imageHeight' in segdict and 'imageWidth' in segdict and (img_hwc.shape[0] != segdict['imageHeight'] or img_hwc.shape[1] != segdict['imageWidth']):
            print("The size of the provided image ({}) does not match the image properties defined in the segmentation file for {}: aborting.".format(Path(img_path).name, segdict['imageFilename']))
            return

        col_msk_hwc = np.zeros( img_hwc.shape, dtype=img_hwc.dtype )
        for reg in segdict['regions']:
            color_count = len(reg['lines'])
            colors = get_n_color_palette( color_count )
            for l,line in enumerate(reg['lines']):
                col = np.array(colors[l % len(colors) ])
                if features['polygons']:
                    rr,cc = (np.array(line['boundary']).T)[::-1]
                    coords = ski.draw.polygon( rr, cc )
                    col_msk_hwc[ coords ] = (col/255.0)
                    bm_hw[ coords ] = True
                    #plt.plot( cc,rr, linewidth=2 )

                if features['baselines'] and 'baseline' in line:
                    baseline_arr = np.sort(np.array( line['baseline'] ), axis=0)
                    plt.plot( baseline_arr[:,0], baseline_arr[:,1], linewidth=2)
            
            if features['regions'] and 'boundary' in reg:
                reg_closed_boundary = np.array( reg['boundary']+[reg['boundary'][0]])
                plt.plot( reg_closed_boundary[:,0], reg_closed_boundary[:,1], linewidth=linewidth)
        col_msk_hwc *= alpha
        bm_hw1 = bm_hw[:,:,None]
        img_complementary = img_hwc * ( ~bm_hw1 + bm_hw1 * (1-alpha))
        composed_img_array = img_complementary + col_msk_hwc

        plt.imshow( composed_img_array )
        plt.title( Path(img_path).name )
        plt.show()


def display_annotated_img( img: Tensor, target: dict, alpha=.4, color='g'):
    """ Overlay of instance masks.
    Args:
        img (Tensor): (C,H,W) image
        target (dict[str,Tensor]): a dictionary of labels with
        - 'masks'=(N,H,W) tensor of masks, where N=# instances for image
        - 'boxes'=(N,4) tensor of BB coordinates
        - 'labels'=(N) tensor of box labels
    """
    img = img.detach().numpy()
    masks = target['masks'].detach().numpy()
    masks = [ m * (m>.5) for m in masks ]
    boxes = [ [ int(c) for c in box ] for box in target['boxes'].detach().numpy().tolist()]
    bm = np.sum( masks, axis=0).astype('bool')
    col = {'r': [1.0,0,0], 'g':[0,1.0,0], 'b':[0,0,1.0]}[color]
    # RED * BOOL * ALPHA
    red_mask = np.transpose( np.full((img.shape[2],img.shape[1],3), col), (2,0,1)) * bm * alpha
    img_complementary = img * ( ~bm + bm * (1-alpha))
    composed_img_array = np.transpose(img_complementary + red_mask, (1,2,0))
    pil_img = Image.fromarray( (composed_img_array*255).astype('uint8'))
    draw = ImageDraw.Draw( pil_img )
    polygon_boundaries = [[ [box[0],box[0]], [box[0],box[1]], [box[1],box[1]], [box[1],box[0]] ] for box in boxes] 
    for i,polyg in enumerate(polygon_boundaries):
        if i%2 != 0:
            draw.polygon(polyg, outline='blue')
    plt.imshow( np.array( pil_img ))
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
        print(len(msks))
        for m in msks[sc>box_threshold]:
            m = m[0]
            plt.imshow( m*(m>mask_threshold) )
            plt.show()
