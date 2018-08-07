import pickle
import time

import numpy
import os
from matplotlib import pyplot
from scipy.interpolate import LinearNDInterpolator
from pyBat import Misc
from acoustics import atmosphere
from pyBat import Frame


def default_atmosphere(temperature=20, relative_humidity=100):
    temperature += 273.15
    return atmosphere.Atmosphere(temperature=temperature, relative_humidity=relative_humidity)


def ratio2db(ratio):
    db = 20 * numpy.log10(ratio)
    return db


def db2ratio(db):
    db = numpy.array(db, dtype='f')
    db = db.astype(float)
    ratio = 10 ** (db / 20.0)
    return ratio


def db2pa(db):
    db = db.astype(float)
    return db2ratio(db) * (2.0 * 10 ** -5)


def incoherent_sum(db_levels):
    db_levels = numpy.array(db_levels, dtype='f')
    db_levels = db_levels[db_levels > 0]
    if db_levels.size == 0: return 0
    summed = numpy.sum(10 ** (db_levels / 10.0))
    summed = 10 * numpy.log10(summed)
    return summed


def first_delay(left, right, delays, threshold=0):
    left_indices = numpy.where(left > threshold)[0]
    right_indices = numpy.where(right > threshold)[0]
    if left_indices.size == 0: left_min = 10000
    if right_indices.size == 0: right_min = 10000
    if left_indices.size > 0: left_min = numpy.min(delays[left_indices])
    if right_indices.size > 0: right_min = numpy.min(delays[right_indices])
    first = numpy.min([left_min, right_min])
    return first


def delay2dist(delay, speed=343):
    delay = numpy.array(delay, dtype='f')
    dist = delay * 0.5 * speed
    return dist


def dist2delay(dist, speed=343):
    dist = numpy.array(dist, dtype='f') * 2
    delay = dist / speed
    return delay


class Sonar:
    def __init__(self, source, frequency, az_rot=0, el_rot=0, dfreq=10000):
        self.atmosphere = default_atmosphere()
        self.emission_db = 120
        self.frequency = frequency
        self.reference_distance = 0.1
        self.reflection_parameter = 0  # Parameter C1 in Stilz and Schnitzler
        self.spreading_parameter = -10  # Parameter C2 in Stilz and Schnitzler
        self.azimuth_rotation = az_rot
        self.elevation_rotation = el_rot
        self.sonar_directivity = Directivity(source, frequency - dfreq, frequency + dfreq, az_rot=az_rot, el_rot=el_rot)
        self.restrict_to_front = True

    @property
    def speed(self):
        return self.atmosphere.soundspeed

    @property
    def attenuation_coefficient(self):
        return self.atmosphere.attenuation_coefficient(self.frequency) * -1

    def spreading(self, distance):
        distance = numpy.array(distance)
        return self.spreading_parameter * numpy.log10(distance / self.reference_distance)

    def attenuation(self, distance):
        distance = numpy.array(distance, dtype='f')
        return self.attenuation_coefficient * distance * 2

    def directivity(self, az, el):
        left, right = self.sonar_directivity.get_directivity(az, el)
        return left, right

    def call_sph(self, az, el, distance):
        distance = numpy.asarray(distance)
        left_db, right_db = self.directivity(az, el)
        gain = self.emission_db + self.spreading(distance) + self.attenuation(distance) + self.reflection_parameter
        gain = numpy.squeeze(gain)
        gain_left = gain + left_db
        gain_right = gain + right_db
        delay = self.dist2delay(distance)
        delay = numpy.asarray(delay)
        if self.restrict_to_front:
            gain_left[numpy.abs(az) > 90] = 0
            gain_right[numpy.abs(az) > 90] = 0
        return gain_left, gain_right, gain, delay

    def call_cart(self, x, y, z):
        az, el, distance = Frame.cart2sph(x, y, z)
        return self.call_sph(az, el, distance)

    def delay2dist(self, delay):
        return delay2dist(delay, self.speed)

    def dist2delay(self, dist):
        return dist2delay(dist, self.speed)


class Directivity:
    def __init__(self, source, freq1, freq2, total=True, az_rot=0, el_rot=0):
        self.left = None
        self.right = None
        self.nose = None
        self.data = None
        self.frequency_range = None
        self.left_function = None
        self.right_function = None
        self.read_pickle(source)
        self.prepare_hrtf(freq1, freq2)
        self.azimuth_rotation = az_rot
        self.elevation_rotation = el_rot
        self.make_functions(total)

    def read_pickle(self, source):
        local = os.path.abspath(__file__)
        dir = os.path.dirname(local)
        name = 'hrtfs/' + source + '.hrtf'
        name = os.path.join(dir, name)
        stream = open(name, 'rb')
        self.data = pickle.load(stream)
        stream.close()

    def prepare_hrtf(self, freq1, freq2):
        freq = self.data['freq']
        left = self.data['left']
        right = self.data['right']
        nose = self.data['nose']

        value1, index1 = Misc.closest(freq, freq1)
        value2, index2 = Misc.closest(freq, freq2)
        index2 += 1

        left = left[:, :, index1:index2]
        right = right[:, :, index1:index2]
        nose = nose[:, :, index1:index2]

        self.frequency_range = freq[index1:index2]
        self.left = numpy.mean(left, 2)
        self.right = numpy.mean(right, 2)
        self.nose = numpy.mean(nose, 2)

    def make_functions(self, total=True):
        if total:
            left = self.left * self.nose
            right = self.right * self.nose
        else:
            left = self.left
            right = self.right
        az, el = Misc.angle_arrays(grid=True)
        az, el, ndist = Frame.rotate_sph(self.azimuth_rotation, self.elevation_rotation, 0, az, el, 1)

        left_db = 20 * numpy.log10(left / numpy.max(left))
        right_db = 20 * numpy.log10(right / numpy.max(right))

        az_flat = az.flatten()
        el_flat = el.flatten()
        left_db = left_db.flatten()
        right_db = right_db.flatten()
        points = numpy.column_stack((az_flat, el_flat))
        self.left_function = LinearNDInterpolator(points, left_db)
        self.right_function = LinearNDInterpolator(points, right_db)

    def get_directivity(self, az_i, el_i):
        shape = az_i.shape
        az_i = az_i.flatten()
        el_i = el_i.flatten()
        az_i = numpy.squeeze(az_i)
        el_i = numpy.squeeze(el_i)
        points = numpy.column_stack((az_i, el_i))
        left_data = self.left_function(points)
        right_data = self.right_function(points)
        left_data = numpy.reshape(left_data, shape)
        right_data = numpy.reshape(right_data, shape)
        return left_data, right_data

    def test_plot(self):
        az_i, el_i = Misc.angle_arrays(step=2.5, grid=True)
        start = time.time()
        left, right = self.get_directivity(az_i, el_i)
        iid = right - left
        az_i, el_i = Misc.angle_arrays(step=2.5, grid=False)

        mono_min = numpy.floor(numpy.nanmin(left))
        iid_min = numpy.floor(numpy.nanmin(iid))

        levels_mono = numpy.arange(mono_min, 3, 3)
        levels_iid = numpy.arange(iid_min, -iid_min + 3, 3)

        end = time.time()
        pyplot.figure(figsize=(15, 5))
        pyplot.subplot(1, 2, 1)
        pyplot.contourf(az_i, el_i, left, levels=levels_mono)
        ax = pyplot.gca()
        ax.set_aspect('equal')
        pyplot.title('Left')
        pyplot.subplot(1, 2, 2)
        pyplot.contourf(az_i, el_i, right, levels=levels_mono)
        ax = pyplot.gca()
        ax.set_aspect('equal')
        pyplot.title('Right')
        pyplot.figure()
        pyplot.contourf(az_i, el_i, iid, levels=levels_iid)
        ax = pyplot.gca()
        ax.set_aspect('equal')
        pyplot.title('IID')
        pyplot.colorbar()
        pyplot.grid()
        pyplot.show()
        print('Interpolation Duration:', end - start)
        return left, right, iid




