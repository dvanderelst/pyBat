import pandas
import os
import numpy
from pyBat import Frame
from matplotlib import pyplot
from scipy.interpolate import interp1d


def read_data(sheet):
    script_path = os.path.realpath(__file__)
    script_path, b = os.path.split(script_path)
    data_path = os.path.join(script_path, 'data', 'Ghose.xls')
    data = pandas.read_excel(data_path, sheetname=sheet)
    return data


def normalize(data):
    # Translate bat data
    Bx = data.Bat_X - data.Bat_X[0]
    By = Bx * 0
    Bz = data.Bat_Z - data.Bat_Z[0]
    # Translate insect data
    Ix = data.Insect_X - data.Bat_X[0]
    Iy = Ix * 0
    Iz = data.Insect_Z - data.Bat_Z[0]
    # Get initial bat heading
    heading, el, rot = Frame.cart2sph(Bx[1], 0, Bz[1])
    # Rotate Insect data
    new_x, new_y, new_z = Frame.rotate_cart(-heading, 0, 0, Ix, Iy, Iz)
    data['Ix'] = new_x
    data['Iy'] = new_y
    data['Iz'] = new_z
    # Rotate Bat Data
    new_x, new_y, new_z = Frame.rotate_cart(-heading, 0, 0, Bx, By, Bz)
    data['Bx'] = new_x
    data['By'] = new_y
    data['Bz'] = new_z
    return data


def plot_data(data):
    pyplot.figure()

    ax = pyplot.subplot(121)
    pyplot.plot(data.Insect_X, data.Insect_Z)
    pyplot.plot(data.Bat_X, data.Bat_Z)
    pyplot.legend(['Insect', 'Bat'])
    ax.set_aspect(1)

    ax = pyplot.subplot(122)
    pyplot.plot(data.Ix, data.Iz)
    pyplot.plot(data.Bx, data.Bz)
    pyplot.legend(['Insect', 'Bat'])
    ax.invert_xaxis()
    ax.set_aspect(1)


class Ghose:
    def __init__(self, data_set):
        data = read_data(data_set)
        meta = read_data(data_set + ' Meta')
        data = normalize(data)

        self.data = data
        self.meta = meta

    def plot_raw(self):
        plot_data(self.data)

    def plot(self, time_stamp=None):
        pyplot.figure()
        x, z = self.get_position(actor='Insect', time_stamp=time_stamp)
        pyplot.scatter(x, z)
        x, z = self.get_position(actor='Bat', time_stamp=time_stamp)
        pyplot.scatter(x, z)
        pyplot.legend(['Insect', 'Bat'])
        pyplot.xlabel('X')
        pyplot.ylabel('Z')
        ax = pyplot.gca()
        ax.invert_xaxis()
        ax.set_aspect(1)
        pyplot.show()

    def get_meta(self, field):
        meta = self.meta[field]
        meta = meta.values
        if len(meta) == 1: meta = float(meta)
        return meta

    def get_distance(self):
        x1, z1 = self.get_position('Bat')
        x2, z2 = self.get_position('Insect')
        distance = ((x1 - x2) ** 2 + (z1 - z2) ** 2) ** 0.5
        return distance

    def get_speed(self, actor='Bat', time_stamp=None):
        delta = 0.01
        time_values = self.data.Time
        if time_stamp is None: time_stamp = time_values
        x1, z1 = self.get_position(actor=actor, time_stamp=time_stamp)
        x2, z2 = self.get_position(actor=actor, time_stamp=time_stamp + delta)
        distance = ((x1 - x2) ** 2.0 + (z1 - z2) ** 2.0) ** 0.5
        speed = distance / delta
        return speed

    def get_position(self, actor='Bat', time_stamp=None):
        time_values = self.data.Time
        if time_stamp is None: time_stamp = time_values
        time_stamp = numpy.array(time_stamp, dtype='f')
        past_values = time_stamp < 0
        time_stamp[past_values] = time_stamp[past_values] + numpy.max(time_values)

        if actor == 'Bat':
            x_values = self.data.Bx.values
            z_values = self.data.Bz.values
        if actor == 'Insect':
            x_values = self.data.Ix.values
            z_values = self.data.Iz.values

        xrange = (x_values[0], x_values[-1])
        zrange = (z_values[0], z_values[-1])

        time_stamp = numpy.array(time_stamp, dtype='f')
        past_values = time_stamp < 0
        time_stamp[past_values] = time_stamp[past_values] + numpy.max(time_values)

        function_x = interp1d(time_values, x_values, fill_value=xrange, bounds_error=False)
        function_z = interp1d(time_values, z_values, fill_value=zrange, bounds_error=False)

        xs = function_x(time_stamp)
        zs = function_z(time_stamp)
        return xs, zs


# def calculate(logger):
#     angles = []
#     gammas = []
#     target_x = logger['target_x']
#     target_z = logger['target_z']
#     bat_x = logger['bat_x']
#     bat_z = logger['bat_z']
#     time = logger['time']
#
#     vx = target_x - bat_x
#     vz = target_z - bat_z
#     dvx = numpy.gradient(vx)
#     dvz = numpy.gradient(vz)
#
#     steps = len(target_x)
#     for index in range(0, steps):
#         dx = bat_x[index] - target_x[index]
#         dz = bat_z[index] - target_z[index]
#         az, el, rot = Rotate.cart2sph(dx, 0, dz)
#         angles.append(az)
#
#         vector1 = [vx[index], vz[index]]
#         vector2 = [dvx[index], dvz[index]]
#         gma = (cosine(vector1, vector2) - 1) * -1
#         gammas.append(gma)
#
#     # Smoothing - in the way I think Ghose smoothed their data
#     angles = Misc.unwrap(angles)
#     time_i = numpy.arange(numpy.min(time), numpy.max(time), 0.1)
#     fnt = interp1d(time, angles)
#     alpha = fnt(time_i)
#     # alpha = savgol_filter(alpha, window_length=11, polyorder=1)
#     delta = numpy.gradient(alpha, edge_order=2)
#     delta = delta / 0.1
#
#     # Interpolate gammas
#     gammas = numpy.array(gammas, dtype='f')
#     fnt = interp1d(time, gammas)
#     gammas = fnt(time_i)
#
#     logger = Logger.Logger()
#     logger.data['time'] = time_i
#     logger.data['alpha'] = alpha
#     logger.data['d_alpha'] = delta
#     logger.data['gamma'] = gammas
#
#     # pyplot.subplot(1, 2, 1)
#     # pyplot.scatter(time, angles)
#     # pyplot.plot(time_i, alpha)
#     # pyplot.subplot(1, 2, 2)
#     # pyplot.scatter(time, numpy.gradient(angles))
#     # pyplot.plot(time_i, delta)
#     # pyplot.show()
#     return logger
