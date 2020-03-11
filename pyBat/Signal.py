import numpy
from matplotlib import pyplot
from scipy.signal import butter, lfilter
from scipy.signal import chirp
from scipy.signal import spectrogram
from scipy.signal import windows
from scipy.signal.windows import hamming

def smooth_signal(signal, samples, window):
    if window == 'box': w = windows.boxcar(samples)
    if window == 'han': w = windows.hann(samples)
    if window == 'flat': w = windows.flattop(samples)
    w = w / numpy.sum(w)
    smoothed = numpy.convolve(signal, w, mode='same')
    return smoothed


def sweep(f1, f2, duration, sample_rate=250000, method='hyperbolic', ramp=5):
    steps = numpy.ceil(duration * sample_rate)
    steps = int(steps)
    time_array = numpy.linspace(0, duration, steps)
    signal = chirp(time_array, f1, duration, f2, method=method)
    if ramp > 0:
        window = signal_ramp(signal.size, ramp)
        signal = signal * window
    return time_array, signal


# def simple_spectrum(signal, sample_rate, norm=False):
#     periodogram = spectrum.WelchPeriodogram(signal, sampling=sample_rate)
#     periodogram.run()
#     frequencies = periodogram.frequencies(sides='onesided')
#     psd = periodogram.psd
#     psd = 10 * numpy.log10(psd)
#     if norm: psd = psd - numpy.max(psd)
#     return frequencies, psd

def simple_spectrogam(x, sample_rate=250000, n=128):
    sample_rate = sample_rate / 1000
    f, t, sxx = spectrogram(x, sample_rate, nperseg=n)
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


def signal_ramp_onesided(n, percent):
    w = signal_ramp(n, percent)
    half = int(n / 2)
    w[0:half] = 1
    return w


def rectify(data):
    new = data.copy()
    new[new < 0] = 0
    return new


class LowPassFilter:
    def __init__(self, cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        self.b, self.a = butter(order, normal_cutoff, btype='low', analog=False)

    def run(self, signal):
        y = lfilter(self.b, self.a, signal)
        return y


class BandBassFilter:
    def __init__(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        self.b, self.a = butter(order, [low, high], btype='band')

    def run(self, signal):
        y = lfilter(self.b, self.a, signal)
        return y

# a = signal_ramp_oneside(100,50)
# pyplot.plot(a)
# pyplot.show()
