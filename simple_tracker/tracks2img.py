import numpy as np
from scipy.ndimage import uniform_filter1d

from .tracking2d import tracking2d


def tracks2img(tracks, img_size, scale=1, mode=None, fps = 1000):

    if mode is None:
        if isinstance(tracks, np.ndarray):
            mode = 'all_in'
        elif isinstance(tracks, (list, tuple)):
            mode = 'tracks'

    # init rendered output image
    img = np.zeros(np.array(img_size)*scale)
    vel = np.zeros(np.array(img_size)*scale)
    
    # catch case where point array empty
    if len(tracks) == 0:
        import warnings
        warnings.warn('No points to track')

        return img, vel

    if mode in ('all_in', 'amplitude'):
        
        # unravel list to numpy array
        coords = np.vstack(tracks)[:, :2] if isinstance(tracks, (list, tuple)) else tracks[:, :2]

        # get integer image coordinates
        coords = np.round(coords*scale).astype('int')

        # remove out-of-grid bubbles (ie. the grid is too small)
        valid = (0 < coords[:, 0]) & (coords[:, 0] < img_size[1]*scale) & (0 < coords[:, 1]) & (coords[:, 1] < img_size[0]*scale)

        # get valid indices and number of duplicates (counts)
        idcs, count = np.unique(coords[valid].T, axis=1, return_counts=True)

        if mode == 'amplitude':
            amplitudes = np.vstack(tracks)[:, -1]
            merged_amplitudes = []
            for i, idx in enumerate(idcs.T):
                mask = np.sum((coords[valid]==idx), 1) == 2
                assert sum(mask) == count[i]
                merged_amplitudes.append(np.mean(amplitudes[valid][mask]))
            merged_amplitudes = np.array(merged_amplitudes)
            merged_amplitudes = (merged_amplitudes - min(merged_amplitudes)) / (max(merged_amplitudes) - min(merged_amplitudes))
            #merged_amplitudes = np.round((merged_amplitudes*255)).astype('int')

            # set pixels in image
            img[idcs[1], idcs[0]] = merged_amplitudes
        else:
            # set pixels in image
            img[idcs[1], idcs[0]] = count

    elif mode in ('tracks', 'vel_z', 'velnorm', 'velmean'):

        # init variables
        min_len = 15
        max_linking_distance = 2
        max_gap_closing = 0
        interp_mode = 'interp' if mode == 'tracks' else 'velocityinterp'

        tracks_result = []
        # split tracks into chunks
        for idx in range(len(tracks)//fps):
            # render based on Hungarian linker
            result, result_interp = tracking2d(tracks[idx*fps:(idx+1)*fps], max_linking_distance=max_linking_distance, max_gap_closing=max_gap_closing, min_len=min_len, scale=1/fps, mode=interp_mode)
            tracks_result.extend(result)

        if mode == 'tracks':
            # single recursive function call
            img, vel = tracks2img(tracks_result, img_size=img_size, scale=scale, mode='all_in') if len(tracks_result) > 0 else (img, vel)

        else:
            # velocity methods
            for i in range(len(tracks_result)):

                # get integer image coordinates
                coords = np.round(tracks_result[i][:, :2]*scale).astype('int')

                if mode == 'velmean':
                    velnorm = tracks_result[i][:, 2]
                else:
                    velnorm = uniform_filter1d(np.linalg.norm(tracks_result[i][:, 2:3], ord=2, axis=1), 10) # velocity is smoothed

                if mode == 'vel_z':
                    # encode the direction of the velocity in positive/negative value
                    velnorm *= np.sign(np.mean(tracks_result[i][:, 2]))

                # remove out of grid bubbles (ie. the grid is too small)
                valid = (0 < coords[:, 0]) & (coords[:, 0] < img_size[1]*scale) & (0 < coords[:, 1]) & (coords[:, 1] < img_size[0]*scale)

                # get valid indices and number of duplicates (counts)
                idcs, count = np.unique(coords[valid].T, axis=1, return_counts=True)

                velnorm = velnorm[valid]

                # set pixels in image
                img[idcs[1], idcs[0]] += count

                # The sum of velocities will be average with Matout at the end of the code.
                vel[idcs[1], idcs[0]] += velnorm[:len(idcs.T)]#, idcs[0]]

    else:
        Exception('Wrong mode selected')

    return img, vel
