import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from simple_tracker.tracking2d import tracking2d
from simple_tracker.tracks2img import tracks2img


if __name__ == '__main__':

    script_path = Path(__file__).parent.resolve()
    data_path = Path('/home/chris/Desktop')   #script_path
    fnames = sorted((data_path / 'output_frames_2ndcomplete').iterdir())

    frames = []
    for fname in fnames:
        if fname.name.startswith('pace'): frames.append(np.loadtxt(fname, delimiter=',', skiprows=1))
    points = np.array(frames)

    framerate = 500
    wavelength = 9.856e-05
    origin = np.array([-72,  16, 0], dtype=int)

    all_pts = np.vstack(points) / wavelength - origin[:2]
    all_img, vel_map = tracks2img(all_pts, img_size=np.array([84, 134]), scale=10, mode='all_in')

    min_len = 15#
    max_linking_distance = 2
    max_gap_closing = 0#

    points = [p / wavelength for p in points]
    tracks_out, tracks_interp = tracking2d(points, max_linking_distance=max_linking_distance, max_gap_closing=max_gap_closing, min_len=min_len, scale=1/framerate, mode='interp')
    shifted_coords = [np.hstack([p[:, :2] - origin[:2], p[:, 2:]]) for p in tracks_out]
    tracks_img, vel_map = tracks2img(shifted_coords, img_size=np.array([84, 134]), scale=10, mode='tracks')#velnorm')#tracks')

    plt.figure()
    plt.imshow(all_img, cmap='gnuplot')#'inferno'
    plt.show()

    plt.imshow(tracks_img, cmap='gnuplot')#'inferno'
    plt.show()

    plt.imshow(vel_map, cmap='plasma')#'inferno'
    plt.show()
