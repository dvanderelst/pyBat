from scipy.signal import chirp
from scipy.signal import spectrogram
from scipy.signal import hamming
from matplotlib import pyplot
import numpy


def sweep(f1, f2, duration, sample_rate=250000, method='hyperbolic', ramp=5):
    steps = numpy.ceil(duration * sample_rate)
    time_array = numpy.linspace(0, duration, steps)
    signal = chirp(time_array, f1, duration, f2, method=method)
    if ramp > 0:
        window = signal_ramp(signal.size, ramp)
        signal = signal * window
    return time_array, signal


def simple_spectrogam(x, sample_rate=250000):
    sample_rate = sample_rate / 1000
    f, t, sxx = spectrogram(x, sample_rate, nperseg=64)
    pyplot.pcolormesh(t, f, sxx)
    pyplot.ylabel('Frequency [kHz]')
    pyplot.xlabel('Time [msec]')
    pyplot.show()


def signal_ramp(n, percent):
    if percent > 49: percent = 49
    length = int(numpy.floor((n * percent) / 100))
    window = hamming(length * 2 + 1)
    window = window - numpy.min(window)
    window = window / numpy.max(window)
    left = window[0:length + 1]
    right = window[length:]
    buffer = numpy.ones(n - 2 * left.size)
    total = numpy.hstack((left, buffer, right))
    return total