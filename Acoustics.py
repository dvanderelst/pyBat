import numpy
from acoustics import atmosphere


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
