import numpy as np


def nearestneighbor_linker(source, target, max_distance: float = np.inf):
   
    n_source_points = source.shape[0]
    n_target_points = target.shape[0]
    
    d = np.zeros([n_source_points, n_target_points])
    
    # Build distance matrix
    for i in range(n_source_points):
        
        # Pick one source point
        current_point = source[i, :]
        
        # Compute square distance to all target points
        diff_coords = target - np.matlib.repmat(current_point, n_target_points, 1)
        square_dist = np.sum(diff_coords**2, 1)
        
        # Store them
        d[i, ...] = square_dist
    
    # Deal with maximal linking distance: we simply mark these links as already
    # treated, so that they can never generate a link.
    d[d > max_distance**2] = np.inf
    
    target_indices = -1 * np.ones(n_source_points, dtype=int)
    target_distances = np.ones(n_source_points) * np.nan
    
    # Parse distance matrix
    while ~np.all(np.isinf(d)):
        
        min_D, closest_targets = np.min(d, axis=1), np.argmin(d, axis=1)#min(D, [], 2) # index of the closest target for each source points
        sorted_index = np.argsort(min_D)
        
        for i in range(sorted_index.size):

            source_index =  sorted_index[i]
            target_index =  closest_targets[sorted_index[i]]
            
            # Did we already assigned this target to a source?
            if np.any(target_index == target_indices):
                
                # Yes, then exit the loop and change the distance matrix to
                # prevent this assignment
                break
                
            else:
                
                # No, then store this assignment
                target_indices[source_index] = target_index
                target_distances[source_index] = np.sqrt(min_D[sorted_index[i]])
                
                # And make it impossible to find it again by putting the target
                # point to infinity in the distance matrix
                d[:, target_index] = np.inf
                # And the same for the source line
                d[source_index, :] = np.inf
                
                if np.all(np.isinf(d)):
                    break
    
    unassigned_targets = np.setdiff1d(list(range(n_target_points)), target_indices)
    
    return target_indices, target_distances, unassigned_targets
