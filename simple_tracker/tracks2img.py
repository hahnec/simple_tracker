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

    if mode == 'all_in':
        
        # unravel list to numpy array
        coords = np.vstack(tracks) if isinstance(tracks, (list, tuple)) else tracks

        # get integer image coordinates
        coords = np.round(coords*scale).astype('int')

        # remove out-of-grid bubbles (ie. the grid is too small)
        valid = (0 < coords[:, 0]) & (coords[:, 0] < img_size[1]*scale) & (0 < coords[:, 1]) & (coords[:, 1] < img_size[0]*scale)

        # get valid indices and number of duplicates (counts)
        idcs, count = np.unique(coords[valid].T, axis=1, return_counts=True)

        # set pixels in image
        img[idcs[1], idcs[0]] = count

    elif mode == 'tracks':

        # init variables
        min_len = 15
        max_linking_distance = 2
        max_gap_closing = 0

        tracks_result = []
        # split tracks into chunks
        for idx in range(len(tracks)//fps):
            # render based on Hungarian linker
            result, result_interp = tracking2d(tracks[idx*fps:(idx+1)*fps], max_linking_distance=max_linking_distance, max_gap_closing=max_gap_closing, min_len=min_len, scale=1/fps, mode='interp')
            tracks_result.extend(result)

        # recursive function call
        img, vel = tracks2img(tracks_result, img_size=img_size, scale=scale, mode='all_in')

    elif mode in ['vel_z', 'velnorm', 'velmean']:

            for i in range(len(tracks)):

                # get integer image coordinates
                coords = np.round(tracks[i][:, :2]*scale).astype('int')

                if mode == 'velmean':
                    velnorm = tracks[i][:, 2]
                else:
                    velnorm = uniform_filter1d(np.linalg.norm(tracks[i][:, 2:3], ord=2, axis=1), 10) # velocity is smoothed

                if mode == 'vel_z':
                    # encode the direction of the velocity in positive/negative value
                    velnorm = velnorm * np.sign(np.mean(tracks[i][:, 2]))

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
