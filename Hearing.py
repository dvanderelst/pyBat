#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:01:57 2017

@author: dieter
"""
import numpy
import pickle
import os
import time
import random

from scipy.spatial.distance import cdist
from scipy.interpolate import RegularGridInterpolator
from acoustics import atmosphere
from pyBat import Frame


def hz2erbrate(f):
    rate = (21.4 * numpy.log10(4.37e-3 * f + 1))
    return rate


def erbrate2hz(r):
    erb = (10 ** (r / 21.4) - 1) / 4.37e-3
    return erb

def erb(f):
    return 24.7 * (4.37 * f/1000 + 1)

def make_erb_cfs(minf, maxf):
    min_rate = numpy.floor(hz2erbrate(minf))
    max_rate = numpy.floor(hz2erbrate(maxf))
    num_channels = max_rate - min_rate
    x = numpy.linspace(min_rate, max_rate, num_channels)
    cfs = erbrate2hz(x)
    return cfs


def q2f(frequency, qvalue):
    delta_f = frequency / qvalue
    return delta_f


def f2q(frequency, delta_f):
    qvalue = frequency / delta_f
    return qvalue


def default_atmosphere(temperature=20, relative_humidity=100):
    temperature += 273.15
    return atmosphere.Atmosphere(temperature=temperature, relative_humidity=relative_humidity)


class Locator:
    def __init__(self, freq1=50000, freq2=90000, source='pd01'):
        self.atmosphere = default_atmosphere()
        self.min_freq = freq1
        self.max_freq = freq2
        self.frequency = (freq1 + freq2) / 2
        self.source_name = source

        self.raw_data = None
        self.transfer_function = None
        self.templates = None
        self.center_frequencies = None
        self.az_array = None
        self.el_array = None

        # acoustic settings
        self.sd_noise = 3
        self.reference_distance = 0.1
        self.reflection_parameter = -20  # Parameter C1 in Stilz and Schnitzler but for 10 cm
        self.spreading_parameter = -40  # Parameter C2 in Stilz and Schnitzler

        self.read_transfer_function()
        self.make_templates()

    @property
    def attenuation_coefficient(self):
        return self.atmosphere.attenuation_coefficient(self.frequency) * -1

    def spreading(self, distance):
        distance = numpy.array(distance, dtype='f')
        return self.spreading_parameter * numpy.log10(distance / self.reference_distance)

    def attenuation(self, distance):
        distance = numpy.array(distance, dtype='f')
        return self.attenuation_coefficient * distance * 2

    def read_transfer_function(self):
        local = os.path.abspath(__file__)
        folder = os.path.dirname(local)
        name = 'hrtfs/' + self.source_name + '.hrtf'
        name = os.path.join(folder, name)
        stream = open(name, 'rb')
        data = pickle.load(stream)
        stream.close()
        self.raw_data = data

    def make_templates(self):
        nose = self.raw_data['nose']
        left = self.raw_data['left']
        freq = self.raw_data['freq']
        az = numpy.arange(-180, 181, 2.5)
        el = numpy.arange(-90, 91, 2.5)

        # make total
        total = nose * left

        # interpolate total
        cf_array = make_erb_cfs(self.min_freq, self.max_freq)
        cf_array = cf_array[cf_array >= numpy.min(freq)]
        cf_array = cf_array[cf_array <= numpy.max(freq)]
        eli = numpy.linspace(-90, 90, 181)
        azi = numpy.linspace(-90, 90, 181)
        extra = numpy.array([180])
        azi = numpy.concatenate((azi, extra))
        az_grid, el_grid, cf_grid = numpy.meshgrid(azi, eli, cf_array)
        interpolator = RegularGridInterpolator((el, az, freq), total)
        total = interpolator((el_grid, az_grid, cf_grid))

        # make templates
        total = total / numpy.max(total)
        log_total_left = 20 * numpy.log10(total)
        log_total_right = numpy.fliplr(log_total_left)

        shape = log_total_left.shape
        template_rows = shape[0] * shape[1]
        template_cols = shape[2]

        left = numpy.reshape(log_total_left, (template_rows, template_cols))
        right = numpy.reshape(log_total_right, (template_rows, template_cols))
        azs, els = numpy.meshgrid(azi, eli)

        self.templates = numpy.concatenate((left, right), axis=1)
        self.center_frequencies = cf_array
        self.az_array = numpy.reshape(azs, (-1, 1))
        self.el_array = numpy.reshape(els, (-1, 1))
        self.az_grid = azs
        self.el_grid = els
        self.transfer_function = {'left': log_total_left, 'right': log_total_right, 'az': az_grid, 'el': el_grid}

    def get_template_sph(self, real_az, real_el, left_only=False):
        template_index = numpy.argmin((self.az_array - real_az) ** 2 + (self.el_array - real_el) ** 2)
        template = self.templates[template_index, :]
        template = numpy.reshape(template, (1, -1))
        length = int(template.shape[1] / 2)
        if left_only: template = template[:, 0:length]
        return template

    def get_template_cart(self, real_x, real_y, real_z, left_only=False):
        real_az, real_el, _ = Frame.cart2sph(real_x, real_y, real_z)
        return self.get_template_sph(real_az, real_el, left_only)

    def locate(self, real_az, real_el, distance, detection_threshold=20):
        start = time.time()
        out_of_fov = False
        if abs(real_az) > 90: out_of_fov = True
        if abs(real_el) > 90: out_of_fov = True

        spreading = self.spreading(distance)
        attenuation = self.attenuation(distance)

        echo_db = 120 + spreading + attenuation + self.reflection_parameter

        templates = self.templates + echo_db
        measurement = self.get_template_sph(real_az, real_el) + echo_db

        #todo: change the detection threshold
        if numpy.max(measurement) < detection_threshold: out_of_fov = True

        # locate if perceivable
        if not out_of_fov:
            shape_templates = templates.shape
            noise = numpy.random.randn(shape_templates[1]) * self.sd_noise
            noisy_measurement = measurement + noise

            templates[templates < detection_threshold] = detection_threshold
            noisy_measurement[noisy_measurement < detection_threshold] = detection_threshold

            dist = cdist(templates, noisy_measurement, 'euclidean')
            index = numpy.argmin(dist)
            perceived_az = self.az_array[index][0]
            perceived_el = self.el_array[index][0]

        # locate if not perceivable
        if out_of_fov:
            perceived_az = random.randint(-90, 90)
            perceived_el = random.randint(-90, 90)

        duration = time.time() - start
        result = {}
        result['echo_db'] = echo_db
        result['spreading'] = spreading
        result['attenuation'] = attenuation
        result['az'] = perceived_az
        result['el'] = perceived_el
        result['duration'] = duration
        result['out_of_fov'] = out_of_fov
        return result

#L = Locator()
