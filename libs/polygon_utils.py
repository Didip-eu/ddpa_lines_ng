import numpy as np
import copy

import shapely


def strip_from_baseline(baseline_n2xy: list[tuple[int,int]], x_height: int, factor: float, ltrb: tuple[int,int,int,int]=tuple()) -> list[tuple[int,int]]:
    """
    Given a baseline, construct the strip-shaped polygon with given height.

    Args:
        baseline_n2xy (list[tuple[int,int]]): a sequence of (x,y) points.
        x_height (float): the line x_height.
        factor (float): scaling factor
        ltrb (tuple[int,int,int,int]): LTRB constraint of containing region: if not empty (default),
            shift coordinates that would otherwise exceed the region's boundaries.
    Returns:
        list[tuple[int,int]]: a (N,2) clockwise sequence of (x,y) points.
    """
    raw_polygon = strip_from_centerline( np.array( baseline_n2xy )-[0,x_height/2], int(x_height*factor) )
    if ltrb:
        return np.clip( raw_polygon, ltrb[:2], ltrb[2:] ).tolist()
    return raw_polygon.tolist()


def strip_from_centerline(centerline_n2xy: np.ndarray, height: float) -> np.ndarray:
    """
    Given a centerline, construct the strip-shaped polygon with given height.

    Args:
        centerline_n2xy (np.ndarray): a (N,2) sequence of (x,y) points.
        height (float): the strip height.
    Returns:
        np.ndarray: a (N,2) clockwise sequence of (x,y) points.
    """
        # degenerate case: centerline is an actual flat line
    if centerline_n2xy.shape==(2,2) and centerline_n2xy[0,1]==centerline_n2xy[1,1]:
        baseline = np.array( centerline_n2xy - int(height/2))
        topline = (baseline+height)[::-1]
        return np.concatenate( (topline, baseline) ).astype('int32')

    left_dummy_pt = np.array( [ 2*centerline_n2xy[0][0]-centerline_n2xy[1][0], 2*centerline_n2xy[0][1]-centerline_n2xy[1][1] ])
    right_dummy_pt = np.array( [ 2*centerline_n2xy[-1][0]-centerline_n2xy[-2][0], 2*centerline_n2xy[-1][1]-centerline_n2xy[-2][1] ])
    centerline_n2xy = np.concatenate( [ [left_dummy_pt], centerline_n2xy, [right_dummy_pt] ], dtype='float')

    vertebras_n2xy = []
    vertebra_north_south_2xy = np.array([[0,-height/2], [0,height/2]])
    for ctr_idx in range(1,len(centerline_n2xy)-1):
        left, mid, right = centerline_n2xy[ctr_idx-1:ctr_idx+2]
        try:
            rotation_matrix = bisection_rotation_matrix( left-mid, right-mid )
            rotated_vertebra_north_south_2xy=np.matmul( rotation_matrix, vertebra_north_south_2xy.T).T
            vertebras_n2xy.append( rotated_vertebra_north_south_2xy + mid ) # shift to actual pos.
        except Exception as e:
            #print(f"Warning: Could not build vertebra from this point: {e}")
            continue
    vertebras_n2xy = np.stack(vertebras_n2xy)
    contour_pts_n2xy = np.concatenate( [vertebras_n2xy[:,0], vertebras_n2xy[::-1,1], vertebras_n2xy[0:1,0]])
    return contour_pts_n2xy.astype('int32')


def bisection_rotation_matrix(left, right):
    """ Given 2 vectors <left> and <right>, return the matrix that rotates a vertical vector 
    such that it bisects the angle formed by <left> and <right>.

    left (float): a 2D vector/pt.
    right (float): a 2D vector/pt.
    """
    # special case (1): vertical segment
    if np.isclose(left[0], right[0]):
        raise ValueError("Vertical segment: abort.")
    # special case (2): colinear, horizontal vectors
    if np.isclose(left[1], right[1]): 
        return np.identity(2)
    alpha, beta, gamma = 0, 0, 0
    if left[0] == 0 and right[0] != 0: # L vector is horizontal
        beta = np.arctan(right[1]/right[0])
        gamma = beta/2
    elif right[0] == 0 and left[0] != 0: # R vector is horizontal
        alpha = np.arctan(left[1]/left[0]) 
        gamma = -alpha/2
    else:
        alpha = np.arctan(left[1]/left[0]) 
        beta = np.arctan(right[1]/right[0]) 
        gamma = ( alpha + beta ) / 2
    cosg, sing = np.cos( gamma ), np.sin( gamma )
    rotation_matrix = np.array([[cosg, -sing],[sing, cosg]])
    return rotation_matrix


def dilate_line_strips( segmentation_dict: dict, step=1, window=4, no_change_max=100, check=False ):
    """Dilate the line strips from their baseline skeleton, while checking for overlaps.
    Line dictionaries are modified in-place.

    TODO: 
    + defense against pathological cases (say, only two lines recognized in region): could be a 
      maximum ratio of actual size to x-height, e.g. 1-to-10
    + offset with respect to centerline, in order to focus on superscript space: safe way is to
    work from (temporary) offset centerlines at the start.
        
    Args:
        segmentation_dict (dict): segmentation dictionary, where each line stores the
            centerline, the x-height, or both.

    Returns:
        dict: segmentation label with original x-heights and updated polygons.
    """
    # get line references
    if 'regions' not in segmentation_dict or len(segmentation_dict['regions']) == 0:
        return segmentation_dict
    segmentation_dict = copy.deepcopy( segmentation_dict )

    lines = list([ l for r in segmentation_dict['regions'] if 'lines' in r for l in r['lines']])
    if not lines:
        return segmentation_dict

    # dummy 
    def offset( cl: np.ndarray, xheight, ratio=.2 ):
        out_cl = np.copy( cl )
        out_cl[:,1] -= int(xheight * ratio)
        return out_cl

    sorted_lines = sorted(lines, key=lambda l: np.array(l["centerline"])[:,1].mean() )
    line_to_height = { l:[ld['x-height'], False] for l,ld in enumerate(sorted_lines)}
    current_line_status, last_change_heights = zip( *line_to_height.values())
    no_change_count = 0

    comparison_matrix = []
    sz = len(sorted_lines)
    for i in range(sz):
        for j in range(0,sz-i):
            if i!=j and abs(i-j)<window:
                comparison_matrix.append( (i,j) )

    while True:
        # construct a polygon for each strip
        polygons = [ shapely.polygons(shapely.linearrings( strip_from_centerline( np.array( ld['centerline'] ), line_to_height[l][0] ))) for l, ld in enumerate( sorted_lines) ]
        for i,j in comparison_matrix:
                #print(f"Testing polygons {i} and {j}")
                if shapely.intersects(polygons[i], polygons[j]):
                    print(f"polygons {i} and {j} intersect")
                    line_to_height[i][1]=True
                    line_to_height[j][1]=True
        
        line_heights, line_status = zip(*line_to_height.values())
        print(''.join([ "T" if t else "F" for t in line_status ]))
        if line_status != current_line_status:
            current_line_status = line_status
            last_change_heights = line_heights
        else:
            no_change_count += 1
            if no_change_count > no_change_max:
                break
        for k,v in line_to_height.items(): 
            if not v[1]:
                line_to_height[k][0] += step

    for k,_ in line_to_height.items():
        line_to_height[k][0] = last_change_heights[k]-step
        sorted_lines[k]['coords'] = strip_from_centerline( np.array( sorted_lines[k]['centerline']), line_to_height[k][0]).tolist()

    # check-only
    if not check:
        return segmentation_dict

    polygons = [ shapely.polygons(shapely.linearrings( strip_from_centerline( np.array( ld['centerline'] ), line_to_height[l][0] ))) for l, ld in enumerate( sorted_lines) ]
    check=True
    for i,j in comparison_matrix:
        if shapely.intersects(polygons[i], polygons[j]):
            check=False

    #print( [line['x-height'] for line in sorted_lines ])
    #print( [line_to_height[l][0] for l in range(len(sorted_lines)) ], check)

    return segmentation_dict

