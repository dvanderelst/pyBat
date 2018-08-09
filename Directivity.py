import os
import pickle
import random
import time
import copy
import numpy
from matplotlib import pyplot
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import RegularGridInterpolator

import Acoustics
import Geometry
import Misc
from Acoustics import default_atmosphere, make_erb_cfs


def read_hrtf(source, freq1, freq2):
    local = os.path.abspath(__file__)
    folder = os.path.dirname(local)
    name = 'hrtfs/' + source + '.hrtf'
    name = os.path.join(folder, name)
    stream = open(name, 'rb')
    data = pickle.load(stream)
    stream.close()

    freq = data['freq']
    left = data['left']
    right = data['right']
    nose = data['nose']

    value1, index1 = Misc.closest(freq, freq1)
    value2, index2 = Misc.closest(freq, freq2)
    index2 += 1

    left = left[:, :, index1:index2]
    right = right[:, :, index1:index2]
    nose = nose[:, :, index1:index2]
    frequency_range = freq[index1:index2]

    hrtf = {}
    hrtf['nose'] = nose
    hrtf['left'] = left
    hrtf['right'] = right
    hrtf['freq'] = frequency_range
    return hrtf


def rotate_hrtf_slice(original_slice, yaw=0, pitch=0, roll=0, plot=True):
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

    if plot: plot_side_by_side(original_slice, new_slice)
    return new_slice


def rotate_hrtf_data(original_data, yaw=0, pitch=0, roll=0, plot=True):
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


def rotate_hrtf(hrtf, yaw=0, pitch=0, roll=0, plot=True):
    left = hrtf['left']
    right = hrtf['right']
    nose = hrtf['nose']

    new_left = rotate_hrtf_data(left, yaw, pitch, roll, False)
    new_right = rotate_hrtf_data(right, yaw, pitch, roll, False)
    new_nose = rotate_hrtf_data(nose, yaw, pitch, roll, False)

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
    min = numpy.min(original)
    max = numpy.max(original)
    levels = numpy.linspace(min, max, 10)

    pyplot.figure()
    pyplot.subplot(1, 2, 1)
    Misc.plot_map(azimuth, elevation, original.flatten(), levels=levels)
    pyplot.title(labels[0])
    pyplot.subplot(1, 2, 2)
    Misc.plot_map(azimuth, elevation, rotated.flatten(), levels=levels)
    pyplot.title(labels[1])


class TransferFunction:
    def __init__(self , source, freq1, freq2, *, yaw=0, pitch=0, roll=0, db=True, collapse=False):
        hrtf = read_hrtf(source, freq1, freq2)
        self.hrtf = rotate_hrtf(hrtf, yaw, pitch, roll, False)
        self.freq = self.hrtf['freq']
        self.collapsed = False

        # Combine emission and hearing
        self.left = self.hrtf['left'] #* self.hrtf['nose']
        self.right = self.hrtf['right'] #* self.hrtf['nose']

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



    def get_templates(self, azimuths, elevations):
        result_left = []
        result_right = []
        for az_point, el_point in zip(azimuths, elevations):
            #az_point = numpy.array(az_point)
            #el_point = numpy.array(el_point)
            coordinate = (el_point, az_point, self.freq)
            if self.collapsed: coordinate = (el_point, az_point)
            left_data = self.left_function(coordinate)
            right_data = self.right_function(coordinate)
            result_left.append(left_data)
            result_right.append(right_data)

        result_left = numpy.squeeze(result_left)
        result_right = numpy.squeeze(result_right)
        return result_left, result_right


tf = TransferFunction('pd01', freq1=28000, freq2=30000, collapse=True, db=True)
r = tf.get_templates([20, 30], [0, 0])
print(r)

tf = TransferFunction('pd01', freq1=29000, freq2=30000, collapse=False, db=True, pitch=0)
r = tf.get_templates([-20, -30], [0, 0])
print(r[0])
print('')
print(r[1])

tf.plot()
pyplot.show()
