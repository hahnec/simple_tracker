import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from pathlib import Path
from tqdm import tqdm

from simple_tracker.linker.hungarian_linker import hungarian_linker
from simple_tracker.linker.nn_linker import nearestneighbor_linker


def simple_tracker(points, max_linking_distance=None, max_gap_closing=None, method=None, print_opt=True):
    
    # default variables
    max_gap_closing         = 3 if max_gap_closing is None else max_gap_closing
    max_linking_distance    = np.inf if max_linking_distance is None else max_linking_distance
    method                  = 'Hungarian' if method is None else method
    
    n_slices = len(points)

    current_slice_index = 0
    row_indices = [[],]*n_slices
    column_indices = [[],]*n_slices
    unmatched_targets = [[],]*n_slices
    unmatched_sources = [[],]*n_slices
    n_cells = [p.shape[0] for p in points]
    
    for i in tqdm(range(n_slices-1), desc='Frame linking', disable=not print_opt):

        source = points[i]
        target = points[i+1]

        # frame to frame linking
        if method.lower() == 'hungarian':
            target_indices , _, unmatched = hungarian_linker(source, target, max_linking_distance)
        elif method.lower() == 'nearestneighbor':
            target_indices , _, unmatched = nearestneighbor_linker(source, target, max_linking_distance)
        unmatched_targets[i+1] = unmatched

        unmatched_sources[i] = np.argwhere(target_indices == -1)
        
        # prepare holders for links in the sparse matrix
        n_links = np.sum(target_indices != -1)
        row_indices[i] = np.ones(n_links) * np.nan
        column_indices[i] = np.ones(n_links) * np.nan
        
        # put it in the adjacency matrix
        index = 0
        for j in range(target_indices.size):
            
            # if we did not find a proper target to link, we skip
            if target_indices[j] == -1:
                continue
            
            # the source line number in the adjacency matrix
            row_indices[i][index] = current_slice_index + j
            
            # the target column number in the adjacency matrix
            column_indices[i][index] = current_slice_index + n_cells[i] + target_indices[j]
    
            index += 1

        current_slice_index = current_slice_index + n_cells[i]    
    
    row_index = np.concatenate(row_indices)
    column_index = np.concatenate(column_indices)
    link_flag = np.ones_like(row_index)
    n_total_cells = sum(n_cells)

    # initialize sparse matrix
    B = scipy.sparse.csr_matrix((np.ones(len(row_index), dtype=bool), (row_index.astype(int), column_index.astype(int))), shape=(n_total_cells, n_total_cells), dtype=bool)
    
    # gap closing
    current_slice_index = 0
    for i in tqdm(range(n_slices-2), desc='Gap closing', disable=not print_opt):

        # find a target and pars over the target that are not part in a link already
        current_target_slice_index = current_slice_index + n_cells[i] + n_cells[i+1]
        
        for j in range(i + 2, min(i +  max_gap_closing, n_slices)):
            
            source = points[i][unmatched_sources[i], :]
            target = points[j][unmatched_targets[j], :]
            
            if (source.size == 0) | (target.size == 0):
                current_target_slice_index = current_target_slice_index + n_cells[j]
                continue
            
            target_indices, _, _ = nearestneighbor_linker(source, target, max_linking_distance)
            
            # put it in the adjacency matrix
            for k in range(target_indices.size):
                
                # if we did not find a proper target to link, we skip
                if target_indices[k] == -1:
                    continue
                
                # the source line number in the adjacency matrix
                row_index = current_slice_index + unmatched_sources[i][k]
                # the target column number in the adjacency matrix
                column_index = current_target_slice_index + unmatched_targets[j][target_indices[k]]
                
                B[row_index, column_index] = True
            
            new_links_target = target_indices != -1
    
            # make linked sources unavailable for further linking
            if unmatched_sources[i].size == new_links_target.size: 
                unmatched_sources[i] = np.array([], dtype=int)
            elif unmatched_sources[i].size > new_links_target.size:
                print('')

            # make linked targets unavailable for further linking
            unmatched_targets[j] = np.delete(unmatched_targets[j], target_indices[new_links_target])
            
            current_target_slice_index = current_target_slice_index + n_cells[j]
        
        current_slice_index = current_slice_index + n_cells[i]
    
    # find columns full of 0s -> means this cell has no source
    cells_without_source = np.where((B.sum(0) > 0) == 0)[-1]
    n_tracks = len(cells_without_source)
    adjacency_tracks = [[],]*n_tracks

    tracks = [[],]*n_tracks
    
    for i in tqdm(range(n_tracks), desc='Tracks', disable=not print_opt):
        
        tmp_holder = np.ones(n_total_cells) * np.nan
        
        target = cells_without_source[i]
        index = 0
        while target is not None:
            tmp_holder[index] = target
            idcs = np.argwhere(B[target, :]).squeeze()[1:]
            target = idcs[0] if idcs.size > 1 else idcs if idcs.size > 0 else None
            index = index + 1

        adjacency_tracks[i] = tmp_holder[~np.isnan(tmp_holder)].astype(int)
    
        # reparse and rectify adjacency track index:
        # The trouble with the previous track index is that the index in each
        # track refers to the index in the adjacency matrix, not the point in
        # the original array
        adjacency_track = adjacency_tracks[i]
        track = np.ones(n_slices) * np.nan
        
        for j in range(adjacency_track.size):
            
            cell_index = adjacency_track[j]
            
            # determine the frame this index belongs to
            tmp = cell_index
            frame_index = 0
            while tmp > 0:
                tmp = tmp - n_cells[frame_index]
                frame_index = frame_index + 1

            frame_index = frame_index - 1
            in_frame_cell_index = tmp + n_cells[frame_index]
            
            track[frame_index] = in_frame_cell_index
        
        tracks[i] = track

    return tracks, adjacency_tracks
