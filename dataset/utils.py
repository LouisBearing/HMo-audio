import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


def affine_matrix(scale_factor, translation_vect, dim):
    if type(scale_factor) != np.ndarray:
        if dim == 3:
            translation_vect = np.concatenate([translation_vect, np.array([0])])
        M = np.hstack((scale_factor * np.eye(dim), np.array(translation_vect).reshape(dim, 1)))
        # Shape dim, 3
    else:
        # Then numpy array
        M = np.concatenate((scale_factor[..., np.newaxis] * (np.eye(dim)[np.newaxis, :]), translation_vect[..., np.newaxis]), axis=-1)
        # Shape 2 * length, dim, 3
    return M


def scale_and_translate(array, M):
    if len(array.shape) == 2:
        array = np.hstack([array, np.ones((len(array), 1))])
        affined_array = np.matmul(array, M.transpose())
    else:
        # Then batch operation
        array = np.concatenate([array, np.ones((array.shape[0], array.shape[1], 1))], axis=-1)
        affined_array = np.matmul(array, M.transpose(0, 2, 1))
    return affined_array


def RyRx_matrix(theta_x, theta_y, theta_z=0):
    rx = R.from_quat([np.sin(theta_x / 2), 0, 0, np.cos(theta_x / 2)]).as_matrix()
    ry = R.from_quat([0, np.sin(theta_y / 2), 0, np.cos(theta_y / 2)]).as_matrix()
    ryrx = np.matmul(ry, rx)
    if theta_z == 0:
        return ryrx
    else:
        rz = R.from_quat([0, 0, np.sin(theta_z / 2), np.cos(theta_z / 2)]).as_matrix()
        return np.matmul(rz, ryrx)


def T_matrix(translation_vect):
    return np.hstack((np.eye(3), np.array(translation_vect).reshape(3, 1)))


def rotate_3D(np_array, theta_y, theta_x, theta_z=0):
    '''
    np_array: shape seq_length, 68, 3
    '''
    origin = np_array.mean(axis=(0, 1))
    centered_arr = scale_and_translate(np_array.reshape(-1, 3), T_matrix(-origin))
    rotated_centered_arr = np.matmul(centered_arr, RyRx_matrix(theta_x, theta_y, theta_z).transpose())
    rotated_arr = scale_and_translate(rotated_centered_arr, T_matrix(origin)).reshape(np_array.shape)
    return rotated_arr


def frontalize(arr):
    '''
    Frontalize independently each time step in a sequence of mouth landmarks.
    Input: 
        np.array arr
        shape seq_len, 20 (mouth landmarks), 3
    Outputs:
        np.array rot_arr, sequence of frontalized mouth landmarks
        same shape

    '''
    vect = (arr[:, 6] - arr[:, 0])
    z_i = vect[:, 2]
    r = np.sqrt(vect[:, 2] ** 2 + vect[:, 0] ** 2)
    rot_arr = []
    for t in np.arange(len(r)):
        theta_y = -np.arcsin(-z_i[t] / r[t])
        rot_arr.append(rotate_3D(arr[[t]], theta_y, 0))
    return np.concatenate(rot_arr)


def collate_vox_lips(list_batch):
    lengths = [(elt[0].size(0), elt[1].size(0)) for elt in list_batch]
    batch = [torch.cat([elt[i] for elt in list_batch], dim=0) for i in range(len(list_batch[0]))]
    return batch + [lengths]


class Vis(object):
    def __init__(self):
        
        self.fig = plt.figure()
        self.init_ax()
        
#         self.colors = {
#             'chin': 'blue',
#             'eyebrow': 'blue',
#             'nose': 'blue',
#             'eyes': 'blue',
#             'outer_lip': 'blue',
#             'innner_lip': 'blue'
#         }
        
        self.colors = {
            'chin': 'green',
            'eyebrow': 'orange',
            'nose': 'blue',
            'eyes': 'red',
            'outer_lip': 'purple',
            'innner_lip': 'pink'
        }

    def init_ax(self):
        self.ax = self.fig.add_subplot()
        self.ax.cla()
        self.ax.set_xlim([0,1])
        self.ax.set_ylim([0,1])
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)

    def update(self, f, coord):
        self.ax.invert_yaxis()
        self.ax.scatter(coord[:, 0], coord[:, 1], linewidths=1)

        #chin
        self.ax.plot(coord[0:17,0],coord[0:17,1],marker='',markersize=5,linestyle='-',color=self.colors['chin'],lw=2)
        #left and right eyebrow
        self.ax.plot(coord[17:22,0],coord[17:22,1],marker='',markersize=5,linestyle='-',color=self.colors['eyebrow'],lw=2)
        self.ax.plot(coord[22:27,0],coord[22:27,1],marker='',markersize=5,linestyle='-',color=self.colors['eyebrow'],lw=2)
        #nose
        self.ax.plot(coord[27:31,0],coord[27:31,1],marker='',markersize=5,linestyle='-',color=self.colors['nose'],lw=2)
        self.ax.plot(coord[31:36,0],coord[31:36,1],marker='',markersize=5,linestyle='-',color=self.colors['nose'],lw=2)
        #left and right eye
        self.ax.plot(coord[36:42,0],coord[36:42,1],marker='',markersize=5,linestyle='-',color=self.colors['eyes'],lw=2)
        self.ax.plot(coord[42:48,0],coord[42:48,1],marker='',markersize=5,linestyle='-',color=self.colors['eyes'],lw=2)
        #outer and inner lip
        self.ax.plot(coord[48:60,0],coord[48:60,1],marker='',markersize=5,linestyle='-',color=self.colors['outer_lip'],lw=2)
        self.ax.plot(coord[60:68,0],coord[60:68,1],marker='',markersize=5,linestyle='-',color=self.colors['innner_lip'],lw=2) 


    def plot_mp4(self, save_path, coords, fps=25):
        length = len(coords)
        f = 0

        metadata = dict(title='01', artist='Matplotlib', comment='motion')
        writer = FFMpegWriter(fps=fps, metadata=metadata)

        with writer.saving(self.fig, save_path, 100):
            for i in range(length):
                self.init_ax()
                self.update(f, coords[i])
                writer.grab_frame()
                plt.pause(0.01)
                f += 1
        plt.close()

    
    def plot_gif(self, save_path, coords, fps=25):
        length = len(coords)
        interval = 1000 / fps
        
        def update_gif(f):
            self.init_ax()
            self.update(f, coords[f])

        ani = FuncAnimation(self.fig, update_gif, frames=length, interval=interval)
        ani.save(save_path, writer='pillow')
        plt.close()


def save_ldk_vid(sequence, save_dir, avg=0):
    
    def moving_avg(a, n):
        '''
        Moving average on axis 1
        '''
        if n == 0:
            return a
        b = np.cumsum(a, axis=0)
        b[n:] = b[n:] - b[:-n]
        return np.concatenate([a[:n - 1], b[n - 1:] / n])
    
    arr = sequence[..., :2] / (sequence.max() + 5)
    arr = moving_avg(arr, avg)
    
    # Saving vid
    vis = Vis()
    # video_path = os.path.join(path, 'viz', f'{filename}.mp4')
    vis.plot_mp4(save_dir, arr, fps=25)
