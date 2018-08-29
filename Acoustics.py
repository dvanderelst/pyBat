import numpy
import math
from acoustics import atmosphere
from scipy.signal import chirp
from scipy.signal import spectrogram
from matplotlib import pyplot
from scipy.special import j1

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


def f2q(frequency, delta_f):
    qvalue = frequency / delta_f
    return qvalue


def q2f(frequency, qvalue):
    delta_f = frequency / qvalue
    return delta_f


def make_erb_cfs(minf, maxf):
    min_rate = numpy.floor(hz2erbrate(minf))
    max_rate = numpy.floor(hz2erbrate(maxf))
    num_channels = max_rate - min_rate
    x = numpy.linspace(min_rate, max_rate, num_channels)
    cfs = erbrate2hz(x)
    return cfs


def hz2erbrate(f):
    rate = (21.4 * numpy.log10(4.37e-3 * f + 1))
    return rate


def erbrate2hz(r):
    erb = (10 ** (r / 21.4) - 1) / 4.37e-3
    return erb


def erb(f):
    return 24.7 * (4.37 * f / 1000 + 1)


def delay2dist(delay, speed=343):
    delay = numpy.array(delay, dtype='f')
    dist = delay * 0.5 * speed
    return dist


def dist2delay(dist, speed=343):
    dist = numpy.array(dist, dtype='f') * 2
    delay = dist / speed
    return delay


def sweep(f0,f1, duration, fs, plot=False):
    delta_t = 1/fs
    t = numpy.arange(0,duration, delta_t)
    n = 64
    signal = chirp(t, f0, duration, f1, method='hyperbolic')
    if plot:
        f, t, Sxx = spectrogram(signal, fs,nperseg=n,noverlap=n-1)
        pyplot.pcolormesh(t, f, Sxx)
        pyplot.ylabel('Frequency [Hz]')
        pyplot.xlabel('Time [sec]')
        pyplot.ylim([f1,f0])
        pyplot.show()
    return signal


def freq2lambda(frequency, speed=343):
    wavelength = speed/frequency
    return wavelength


def leaf_coefficient(radius, frequency, angle):
    radius = numpy.array(radius)
    frequency = numpy.array(frequency)
    angle = numpy.array(angle)
    wavelength = freq2lambda(frequency)
    k = (2 * numpy.pi) / wavelength
    radians = numpy.deg2rad(angle)
    a = 0.5 * (k * radius)**2 + 0.7
    b = 0.4 * (k * radius)**-0.9 + 1
    coefficient = a * numpy.cos(b * radians)
    return coefficient

# radius = numpy.array([0.05])
# frequency = numpy.array([70000])
# angle = numpy.array([0])
#
# radians = numpy.deg2rad(angle)
#
# wavelength = freq2lambda(frequency)
#
# k = 2 * numpy.pi / wavelength
#
# beta = 2 * k * radius * numpy.sin(radius)
#
#
# part1 = (numpy.pi * radius ** 2) / wavelength
#
# part2 = (2 * j1(beta) / beta)**2
# part3 = numpy.cos(radians) ** 2
#
# total = part1 * part2 * part3
#
# check1 = radius**2/wavelength
