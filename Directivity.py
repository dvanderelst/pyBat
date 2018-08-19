import copy
import os
import pickle

import numpy
from matplotlib import pyplot
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import RegularGridInterpolator

import Geometry
import Misc


def read_hrtf(source, freq_list):
    local = os.path.abspath(__file__)
    folder = os.path.dirname(local)

    cache_file = 'hrtfs/cache.hrtf'
    hrtf_file = 'hrtfs/' + source + '.hrtf'
    hrtf_file = os.path.join(folder, hrtf_file)
    cache_file = os.path.join(folder, cache_file)
    # Try using cached version - to speed up things
    try:
        stream = open(cache_file, 'rb')
        hrtf = pickle.load(stream)
        stream.close()

        same_freqs = numpy.min(freq_list == hrtf['freq'])
        same_source = source == hrtf['source']
        if same_freqs and same_source: return hrtf
        print('read_hrtf: Reading HRTF cache failed. Loading...')
    except:
        print('read_hrtf: Reading HRTF cache failed. Loading...')

    stream = open(hrtf_file, 'rb')
    data = pickle.load(stream)
    stream.close()

    left = data['left']
    right = data['right']
    nose = data['nose']

    grids = Misc.angle_arrays(grid=False)
    azimuths = grids[0]
    elevations = grids[1]
    frequencies = data['freq']

    coordinates = (elevations, azimuths, frequencies)
    left_function = RegularGridInterpolator(coordinates, left)
    right_function = RegularGridInterpolator(coordinates, right)
    nose_function = RegularGridInterpolator(coordinates, nose)

    n = len(freq_list)
    new_left = numpy.zeros((73, 145, n))
    new_right = numpy.zeros((73, 145, n))
    new_nose = numpy.zeros((73, 145, n))

    for r in range(0, 73):
        for c in range(0, 145):
            el = elevations[r]
            az = azimuths[c]
            interpolation = (el, az, freq_list)
            left_data = left_function(interpolation)
            right_data = right_function(interpolation)
            nose_data = nose_function(interpolation)

            new_left[r, c, :] = left_data
            new_right[r, c, :] = right_data
            new_nose[r, c, :] = nose_data

    hrtf = {}
    hrtf['source'] = source
    hrtf['nose'] = new_nose
    hrtf['left'] = new_left
    hrtf['right'] = new_right
    hrtf['freq'] = freq_list

    f = open(cache_file, 'wb')
    pickle.dump(hrtf, f)
    f.close()

    return hrtf


def rotate_hrtf_slice(original_slice, yaw=0, pitch=0, roll=0, plot=True):
    summed_angles = abs(yaw) + abs(pitch) + abs(roll)
    if summed_angles > 0:
        shape = original_slice.shape
        original_slice = original_slice.flatten()
        grids = Misc.angle_arrays(grid=True)
        azimuth = grids[0].flatten()
        elevation = grids[1].flatten()
        old_dist = numpy.ones(azimuth.shape)

        old_x, old_y, old_z = Geometry.sph2cart(azimuth, elevation, old_dist)
        new_x, new_y, new_z = Geometry.rotate_points_cart(old_x, old_y, old_z, yaw=yaw, pitch=pitch, roll=roll)

        old_locations = numpy.row_stack((old_x, old_y, old_z))
        old_locations = numpy.transpose(old_locations)

        new_locations = numpy.row_stack((new_x, new_y, new_z))
        new_locations = numpy.transpose(new_locations)

        function_inter = NearestNDInterpolator(new_locations, original_slice)
        new_slice = function_inter(old_locations)
        new_slice = numpy.reshape(new_slice, shape)
    else:
        new_slice = copy.copy(original_slice)
    if plot: plot_side_by_side(original_slice, new_slice)
    return new_slice


def rotate_hrtf_block(original_data, yaw=0, pitch=0, roll=0, plot=True):
    new_data = numpy.zeros(original_data.shape)
    n_slice = new_data.shape[2]
    for i in range(0, n_slice):
        slice = original_data[:, :, i]
        new_slice = rotate_hrtf_slice(slice, yaw, pitch, roll, False)
        new_data[:, :, i] = new_slice
    if plot: plot_side_by_side(original_data, new_data)
    return new_data


def normalize_hrtf_data(data, db=False):
    data = copy.copy(data)
    if len(data.shape) == 2: data = numpy.expand_dims(data, axis=2)
    n_slice = data.shape[2]
    for i in range(0, n_slice):
        slice = data[:, :, i]
        slice = slice / numpy.max(slice)
        if db: slice = 20 * numpy.log10(slice)
        data[:, :, i] = slice
    data = numpy.squeeze(data)
    return data


def process_rotation_settings(**kwargs):
    yaw_left = kwargs.pop('yaw_left', None)
    yaw_right = kwargs.pop('yaw_right', None)
    yaw_nose = kwargs.pop('yaw_nose', None)

    pitch_left = kwargs.pop('pitch_left', None)
    pitch_right = kwargs.pop('pitch_right', None)
    pitch_nose = kwargs.pop('pitch_nose', None)

    roll_left = kwargs.pop('roll_left', None)
    roll_right = kwargs.pop('roll_right', None)
    roll_nose = kwargs.pop('roll_nose', None)

    yaw = kwargs.pop('yaw', 0)
    pitch = kwargs.pop('pitch', 0)
    roll = kwargs.pop('roll', 0)

    if yaw_left is None: yaw_left = yaw
    if yaw_right is None: yaw_right = yaw
    if yaw_nose is None: yaw_nose = yaw

    if pitch_left is None: pitch_left = pitch
    if pitch_right is None: pitch_right = pitch
    if pitch_nose is None: pitch_nose = pitch

    if roll_left is None: roll_left = roll
    if roll_right is None: roll_right = roll
    if roll_nose is None: roll_nose = roll

    left = (yaw_left, pitch_left, roll_left)
    right = (yaw_right, pitch_right, roll_right)
    nose = (yaw_nose, pitch_nose, roll_nose)

    return left, right, nose


def rotate_hrtf(hrtf, **kwargs):
    plot = kwargs.pop('plot', False)
    left_rot, right_rot, nose_rot = process_rotation_settings(**kwargs)

    left = hrtf['left']
    right = hrtf['right']
    nose = hrtf['nose']

    yaw, pitch, roll = left_rot
    new_left = rotate_hrtf_block(left, yaw, pitch, roll, False)
    yaw, pitch, roll = right_rot
    new_right = rotate_hrtf_block(right, yaw, pitch, roll, False)
    yaw, pitch, roll = nose_rot
    new_nose = rotate_hrtf_block(nose, yaw, pitch, roll, False)

    if plot:
        plot_side_by_side(left, new_left)
        plot_side_by_side(right, new_right)
        plot_side_by_side(nose, new_nose)

    hrtf['left'] = new_left
    hrtf['right'] = new_right
    hrtf['nose'] = new_nose
    return hrtf


def plot_side_by_side(original, rotated, labels=[]):
    shape = original.shape
    if len(shape) == 3:
        original = numpy.mean(original, axis=2)
        rotated = numpy.mean(rotated, axis=2)

    if len(labels) == 0: labels = ['Original', 'Rotated']

    grids = Misc.angle_arrays(grid=True)
    azimuth = grids[0].flatten()
    elevation = grids[1].flatten()
    min = numpy.min(numpy.minimum(original, rotated))
    max = numpy.max(numpy.maximum(original, rotated))
    levels = numpy.linspace(min, max, 10)

    pyplot.figure()
    pyplot.subplot(2, 1, 1)
    Misc.plot_map(azimuth, elevation, original.flatten(), levels=levels)
    pyplot.title(labels[0])
    pyplot.subplot(2, 1, 2)
    Misc.plot_map(azimuth, elevation, rotated.flatten(), levels=levels)
    pyplot.title(labels[1])


class TransferFunction:
    def __init__(self, source, freq_list, db=True, collapse=False, full=True, **kwargs):
        hrtf = read_hrtf(source, freq_list)
        self.hrtf = rotate_hrtf(hrtf, **kwargs)
        self.freq = self.hrtf['freq']
        self.collapsed = False

        if full:
            self.left = self.hrtf['left'] * self.hrtf['nose']
            self.right = self.hrtf['right'] * self.hrtf['nose']
        else:
            self.left = self.hrtf['left']
            self.right = self.hrtf['right']

        # Collapsing data
        if collapse:
            self.left = numpy.sum(self.left, axis=2)
            self.right = numpy.sum(self.right, axis=2)
            self.collapsed = True

        # Convert to db
        if db:
            self.left = normalize_hrtf_data(self.left, db=True)
            self.right = normalize_hrtf_data(self.right, db=True)

        # Set up interp functions
        grids = Misc.angle_arrays(grid=False)
        azimuth = grids[0]
        elevation = grids[1]
        coordinates = (elevation, azimuth, self.freq)
        if self.collapsed: coordinates = (elevation, azimuth)
        self.left_function = RegularGridInterpolator(coordinates, self.left)
        self.right_function = RegularGridInterpolator(coordinates, self.right)

    def plot(self):
        labels = ['Right', 'Left']
        plot_side_by_side(self.right, self.left, labels)

    def query_collapsed(self, azimuths, elevations):
        azimuths = numpy.array(azimuths)
        elevations = numpy.array(elevations)
        coordinates = numpy.row_stack((elevations,azimuths))
        coordinates = numpy.transpose(coordinates)
        left_data = self.left_function(coordinates)
        right_data = self.right_function(coordinates)
        return left_data, right_data

    def query(self, azimuths, elevations):
        result_left = []
        result_right = []
        if self.collapsed: return self.query_collapsed(azimuths, elevations)
        for az_point, el_point in zip(azimuths, elevations):
            coordinate = (el_point, az_point, self.freq)
            if self.collapsed: coordinate = (el_point, az_point)
            left_data = self.left_function(coordinate)
            right_data = self.right_function(coordinate)
            result_left.append(left_data)
            result_right.append(right_data)

        result_left = numpy.squeeze(result_left)
        result_right = numpy.squeeze(result_right)
        return result_left, result_right


if __name__ == "__main__":
    tf = TransferFunction('pd01', freq_list=[30123], collapse=True, db=True)
    r = tf.query([20, 30], [0, 0])
    # print(r)

    # tf = TransferFunction('pd01', freq1=29, freq2=30, collapse=True, db=True, yaw_left=-10, yaw_right=10)
    # r = tf.get_templates([0, 0], [-10, 20])
    # print('left', r[0])
    # print('')
    # print('right', r[1])

    tf.plot()
    pyplot.show()
