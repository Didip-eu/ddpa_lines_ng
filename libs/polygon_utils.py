import numpy as np


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
        return boxed_in( raw_polygon, ltrb ).tolist()
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
        return np.concatenate( (topline, baseline) )

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
            logger.warning(e)
            continue
    vertebras_n2xy = np.stack(vertebras_n2xy)
    contour_pts_n2xy = np.concatenate( [vertebras_n2xy[:,0], vertebras_n2xy[::-1,1], vertebras_n2xy[0:1,0]])
    return contour_pts_n2xy.astype('int32')


def boxed_in( sequence_n2xy: np.ndarray, ltrb: tuple[float,float,float,float] )->np.ndarray:
    """
    Given a sequence of points, shift its elements' coordinates  s.t. they are contained
    within the given box. Can be used for (y,x) points: be sure to pass the box as (t,l,b,r).

    Args: 
        sequence_n2xy (np.ndarray) a (N,2) sequence of (x,y) points.
        ltrb (tuple[float,float,float,float]): the left, top, right, and bottom coordinates.
    Returns:
        polyg_n2xy (np.ndarray): a (N,2) sequence of (x,y) points.
    """
    left, top, right, bottom = ltrb
    shifted_pts = []
    for pt in sequence_n2xy:
        x, y = pt
        if x < left:
            x = left
        elif x > right:
            x = right
        if y < top:
            y = top
        elif y > bottom:
            y = bottom
        shifted_pts.append( [x,y] )
    return np.array( shifted_pts )


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


