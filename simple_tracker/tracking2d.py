import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d

from simple_tracker.simple_tracker import simple_tracker


def tracking2d(points, max_linking_distance, max_gap_closing, min_len=None, scale=None, mode=None):

    min_len = 15 if min_len is None else min_len
    scale = 1 if scale is None else scale
    mode = 'nointerp' if mode is None else mode

    tracks, adjacency_tracks = simple_tracker(points, max_linking_distance, max_gap_closing)
    
    all_points = np.vstack(points)
    points = [p[None, :] if len(p.shape) == 1 else p for p in points] # add dimension for single point frames
    all_points_fidx = np.vstack([np.hstack([f, i*np.ones([len(f), 1])]) for i, f in enumerate(points)])

    track_id_limit = all_points_fidx.shape[0]

    count=0
    tracks_raw = []
    for i in range(len(tracks)):
        track_id = adjacency_tracks[i]
        if not np.all(track_id < track_id_limit):
            continue
        frame_id = all_points_fidx[track_id, 2].astype(int)   # get frame number of track id
        track_points = np.hstack([all_points_fidx[track_id, :2], frame_id[:, None]])#np.vstack([all_points[track_id, :], idFrame])#np.hstack(all_points[track_id, :], idFrame)
        if len(track_points[:, 0]) > min_len:
            tracks_raw.append(track_points)
            count += 1

    if count==0:
        print('No tracks found')
        return [], []

    smooth_factor = 20#19#
    resolution_factor = 10 # typically 10 for images at lambda/10

    # post processing of tracks
    interp_factor = 1 / max_linking_distance / resolution_factor * .8
    tracks_out = []
    tracks_interp = []
    for i in range(len(tracks_raw)):
        track_points = tracks_raw[i].astype('float')
        xi = track_points[:, 1]
        zi = track_points[:, 0]
        if mode.lower() == 'nointerp':
            # without interpolation, raw tracks
            i_frame = track_points[:, 2]
            if len(zi) > min_len:
                tracks_out.append(np.stack([zi, xi, i_frame]).T)
        if mode.lower() == 'interp':
            # with tracks interpolation
            fun = interp1d(np.arange(0, len(zi)), uniform_filter1d(zi, smooth_factor))
            zu = fun(np.arange(0, len(zi)-1, interp_factor))
            fun = interp1d(np.arange(0, len(xi)), uniform_filter1d(xi, smooth_factor))
            xu = fun(np.arange(0, len(xi)-1, interp_factor))
            if len(zi) > min_len:
                tracks_out.append(np.stack([zu, xu]).T)
        if mode.lower() == 'velocityinterp':
            # with tracks interpolation
            TimeAbs = np.arange(0, (len(zi))) * scale

            # interpolation of spatial and time components
            fun = interp1d(np.arange(0, len(zi)), uniform_filter1d(zi, smooth_factor))
            zu = fun(np.arange(0, len(zi)-1, interp_factor))
            fun = interp1d(np.arange(0, len(xi)), uniform_filter1d(xi, smooth_factor))
            xu = fun(np.arange(0, len(xi)-1, interp_factor))
            fun = interp1d(np.arange(0, len(TimeAbs)), TimeAbs)
            TimeAbs_interp = fun(np.arange(0, len(TimeAbs)-1, interp_factor))

            # velocity
            vzu = np.diff(zu) / np.diff(TimeAbs_interp)
            vxu = np.diff(xu) / np.diff(TimeAbs_interp)
            vzu = np.hstack([vzu[0], vzu])
            vxu = np.hstack([vxu[0], vxu])

            if len(zi)> min_len:
                tracks_out.append(np.stack([zu.T, xu.T, vzu.T, vxu.T, TimeAbs_interp.T]).T) #position / velocity / timeline

        if mode.lower() == 'pala':
        # with and without interpolation, dedicated to PALA comparison of localization algorithms.
            i_frame = track_points[:, 2]

            if len(zi)> min_len:
                # store in Tracks position and frame number, used to compare with
                # simulation dataset where absolution positions are available.
                tracks_out.append(np.stack([zi,xi,i_frame]).T)

            # Interpolate tracks for density rendering
            fun = interp1d(np.arange(0, len(zi)), uniform_filter1d(zi, smooth_factor))
            zu = fun(np.arange(0, len(zi)-1, interp_factor))
            fun = interp1d(np.arange(0, len(xi)), uniform_filter1d(xi, smooth_factor))
            xu = fun(np.arange(0, len(xi)-1, interp_factor))

            dd = np.sqrt(np.diff(xu)**2+np.diff(zu)**2) # curvilinear abscissa
            vmean = np.sum(dd)/len(zi) / scale # averaged velocity of the track in [unit]/s

            if len(zi) > min_len and False:
                tracks_interp.append(np.stack([zu.T,xu.T,vmean*np.ones(zu.T.shape)]).T)

    return tracks_out, tracks_interp
