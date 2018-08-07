import numpy
from matplotlib import pyplot
from pyfilterbank.gammatone import GammatoneFilterbank
from  scipy.signal import butter
from scipy.signal import freqz
from scipy.signal import lfilter

def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def rectify(data):
    new = data.copy()
    new[new < 0] = 0
    return new


class ModelWiegrebe:
    def __init__(self, sample_rate, center, bands):
        self.sample_rate = sample_rate
        self.center = center
        self.bands = bands
        self.gammatone = GammatoneFilterbank(samplerate=sample_rate, startband=-bands, endband=bands, normfreq=center)
        if numpy.max(self.frequencies) > (sample_rate / 2) * 0.9: raise ValueError('Max frequency too high')
        self.b, self.a = butter_lowpass(1000, sample_rate, 2)
        self.signal = None
        self.gamma_output = None
        self.rectified = None
        self.compressed = None
        self.filtered = None
        self.result = None

    @property
    def frequencies(self):
        return self.gammatone.centerfrequencies

    def run_gammatone(self):
        analyse = self.gammatone.analyze(self.signal)
        n = self.signal.size
        bm = numpy.zeros((0, n))
        for band in analyse:
            response = numpy.real(band[0])
            bm = numpy.vstack((bm, response))
        return bm

    def run_filter(self, data):
        output = lfilter(self.b, self.a, data, axis=1)
        return output

    def run_model(self, signal):
        self.signal = signal
        self.gamma_output = self.run_gammatone()
        self.rectified = rectify(self.gamma_output)
        self.compressed = numpy.power(self.rectified, 0.4)
        self.filtered = self.run_filter(self.compressed)
        self.result = rectify(self.filtered)
        return self.result

    def plot_filter(self):
        w, h = freqz(self.b, self.a)
        f = 0.5 * self.sample_rate * w / numpy.pi
        db = 20 * numpy.log10(abs(h))
        pyplot.plot(f, db)