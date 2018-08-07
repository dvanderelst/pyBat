import math

import numpy
from skimage.morphology import binary_opening
from skimage.morphology import disk
from skimage.transform import resize

from pyBat import Misc
from pyBat import Frame


def circle(radius, delta_angle):
    azimuths = numpy.arange(-180, 180, delta_angle)
    x, y, z = Frame.sph2cart(azimuths, 0, radius)
    return x, y, z


def patchy_world(radius, percent, factor=2):
    diameter = int(radius * 2)
    resize_dimension = math.ceil(diameter / factor)
    surface = numpy.random.rand(resize_dimension, resize_dimension)
    surface = resize(surface, (diameter, diameter), order=1)

    element = disk(1)
    selected = 0
    for x in range(0, 100):
        threshold = numpy.percentile(surface, 100 - x)
        selected = surface > threshold
        selected = binary_opening(selected, selem=element)
        occupied = 100 * numpy.mean(selected)
        if occupied >= percent: break

    y, x = numpy.where(selected)
    x = x - (diameter / 2)
    y = y - (diameter / 2)

    distance = numpy.sqrt(x ** 2 + y ** 2)
    in_range = distance < radius
    x = x[in_range]
    y = y[in_range]

    distance = numpy.sqrt(x ** 2 + y ** 2)
    in_range = distance > 5

    x = x[in_range]
    z = y[in_range]
    y = x * 0

    cx, cy, cz = circle(radius, 1)
    x = numpy.hstack((x, cx))
    y = numpy.hstack((y, cy))
    z = numpy.hstack((z, cz))
    return x, y, z


def world_of_dots(radius, seeds, points, sigma, y_zero=True):
    sigmas = [sigma] * 3
    correlations = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    cov = Misc.corr2cov(sigmas, correlations)
    while True:
        x, y, z = Misc.random_cds(seeds * 2, -radius, radius, y_zero)
        distance = numpy.sqrt(x ** 2 + y ** 2 + z ** 2)
        selected = distance < radius
        if numpy.sum(selected) >= seeds: break
    indices = numpy.where(selected)
    indices = indices[0]
    indices = numpy.random.choice(indices, seeds)
    x = x[indices]
    y = y[indices]
    z = z[indices]

    cds = numpy.column_stack((x, y, z))
    all_points = numpy.empty((0, 3))
    for index in range(0, seeds):
        center = cds[index, :]
        generated = numpy.random.multivariate_normal(center, cov, points)
        all_points = numpy.vstack((all_points, generated))

    x = all_points[:, 0]
    y = all_points[:, 1]
    z = all_points[:, 2]

    distance = numpy.sqrt(x ** 2 + y ** 2 + z ** 2)
    selected = distance < radius

    x = x[selected]
    y = y[selected]
    z = z[selected]

    distance = numpy.sqrt(x ** 2 + y ** 2 + z ** 2)
    selected = distance > 5

    x = x[selected]
    y = y[selected]
    z = z[selected]

    if y_zero: y *= 0

    return x, y, z
