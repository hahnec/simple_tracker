import unittest
import numpy as np
import scipy
import matplotlib.pyplot as plt
from pathlib import Path

from simple_tracker.simple_tracker import simple_tracker


class SimpleTrackerTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(SimpleTrackerTest, self).__init__(*args, **kwargs)

    def setUp(self):

        np.random.seed(3006)

        self.plot_opt = False
        self.matlab_data_opt = True
        self.tracks, self.adjacency_tracks = [], []

        # dimensionality of the simulated problem (2 for 2D, 3 for 3D)
        n_dim = 2

        # number of rames to track the points over
        n_frames = 20

        # approximative number of points per frame
        n_points_per_frame = 10

        # create random points
        self.points = []

        # random start position
        start = 20 * np.random.rand(n_points_per_frame, n_dim)

        # span initial direction
        theta = np.linspace(0, 2* np.pi/4, n_points_per_frame).T
        vec = np.array([np.cos(theta), np.sin(theta)]).T

        # random direction change
        theta_increase = np.pi / n_frames * np.random.rand(n_points_per_frame, 1)

        for i_frame in range(n_frames):

            # disperse points as if their position was increasing by 1.5 in average each frame
            frame_points = start + vec * i_frame * np.hstack([np.cos(theta_increase * i_frame), np.sin(theta_increase * i_frame)]) + np.random.rand(n_points_per_frame, n_dim)

            # randomize them
            randomizer = np.random.rand(n_points_per_frame)
            sorted, index = np.sort(randomizer), np.argsort(randomizer)
            frame_points = frame_points[index, :]

            # delete some of them, possible
            deleter = np.random.randn(1)
            while (deleter > 0):
                frame_points = np.delete(frame_points, 0, 0)
                deleter = deleter - 1

            self.points.append(frame_points)
        
        script_path = Path(__file__).parent.resolve()
        if self.matlab_data_opt:
            points_fname = 'test_points.mat'
            points_mat = scipy.io.loadmat(script_path.parent / 'simple_tracker' / 'data' / points_fname)
            self.points = [p[0] for p in points_mat['points']]
        else:
            points_fname = 'py_points.mat'
            scipy.io.savemat(script_path.parent / 'simple_tracker' / 'data' / points_fname, {'points': np.array(self.points, dtype=object)})

    def test_simple_tracker(self):

        max_linking_distance = 4
        max_gap_closing = np.inf
        method = 'Hungarian'

        import time
        start = time.time()

        self.tracks, self.adjacency_tracks = simple_tracker(self.points, max_linking_distance, max_gap_closing, method)

        print(time.time() - start)

        if self.plot_opt: self.plot_results()

    def plot_results(self):

        plt.figure()
        for i_frame in range(len(self.points)):
            i_str = str(i_frame)
            for j_point in range(len(self.points[i_frame])):
                pos = self.points[i_frame][j_point, :]
                plt.plot(pos[0], pos[1], 'x')
                plt.text(*pos, s=i_str)

        all_points = np.vstack(self.points)

        colors = plt.cm.jet(np.linspace(0, 1, len(self.tracks)))

        for i_track in range(len(self.tracks)):
        
            # use the adjacency tracks to retrieve the point coordinates
            track = self.adjacency_tracks[i_track].astype(int)
            track_points = all_points[track, :]
            
            plt.plot(track_points[:, 0], track_points[:, 1], color=colors[i_track, :])

        plt.show()

if __name__ == '__main__':
    unittest.main()