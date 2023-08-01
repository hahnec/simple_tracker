import numpy as np
import numpy.matlib


def hungarian_linker(source, target, max_distance: float = np.inf, method: str = 'scipy'):

    if len(source.shape) == 1:
        n_source_points = 1
        source = source[None, :]
    else:
        n_source_points = source.shape[0]
    n_target_points = target.shape[0]
    
    cost_mat = np.ones([n_source_points, n_target_points]) * np.nan

    # build cost matrix
    for i in range(n_source_points):
        
        # pick one source point
        current_point = source[i, :]
        
        # compute square distance to all target points
        diff_coords = target - np.matlib.repmat(current_point, n_target_points, 1)
        square_dist = np.sum(diff_coords**2, 1)

        # Store them
        cost_mat[i, :] = square_dist
    
    # maximum linking distance: we simply mark these links as already treated, so that they can never generate a link
    cost_mat[cost_mat > max_distance**2] = 10**(np.ceil(np.log10(np.sum(cost_mat)))+1)

    # pad rectangular matrix
    cost_mat = squarify(cost_mat, np.max(cost_mat)) if cost_mat.shape[0] > cost_mat.shape[1] else cost_mat
    
    # find assignment
    if method == 'lapsolver':
        from lapsolver import solve_dense
        _, target_indices = solve_dense(cost_mat)
    elif method == 'scipy':
        from scipy.optimize import linear_sum_assignment
        _, target_indices = linear_sum_assignment(cost_mat)
    elif method == 'munkres':
        from munkres import Munkres
        _, target_indices = [np.array(el) for el in list(zip(*Munkres().compute(cost_mat.tolist())))]

    # set unmatched sources to -1
    unmatched_idcs = np.argwhere([sum(r>max_distance**2) == cost_mat.shape[1] for r in cost_mat])#.squeeze()
    for unmatched_idx in unmatched_idcs.tolist():
        if target_indices.size == cost_mat.shape[0]:
            target_indices[unmatched_idx] = -1
        else:
            target_indices = np.insert(target_indices, unmatched_idx, -1)

    # set unmatched assignments for columns with maximum cost
    valid_idcs = target_indices[target_indices>0]
    valid_mat = cost_mat != cost_mat.max()
    valid_idcs[~valid_mat[target_indices>0, valid_idcs]] = -1
    target_indices[target_indices>0] = valid_idcs

    assert target_indices.size == cost_mat.shape[0], 'mismatch'
    
    # collect distances
    target_distances = np.ones(target_indices.size) * np.nan
    for i in range(target_indices.size):
        if target_indices[i] < 0:
            continue
        
        target_distances[i] = (cost_mat[i, target_indices[i]])**.5

    unassigned_targets = np.setdiff1d(list(range(n_target_points)), target_indices)

    return target_indices, target_distances, unassigned_targets


def squarify(M,val):
    (a,b)=M.shape
    if a>b:
        padding=((0,0),(0,a-b))
    else:
        padding=((0,b-a),(0,0))
    return numpy.pad(M,padding,mode='constant',constant_values=val)