import numpy
from matplotlib import pyplot

from pyBat.Signal import LowPassFilter, rectify
from pyfilterbank.gammatone import GammatoneFilterbank
from scipy.signal import freqz


class ModelWiegrebe:
    def __init__(self, sample_rate, center, bands):
        self.sample_rate = sample_rate
        self.center = center
        self.bands = bands
        self.gammatone = GammatoneFilterbank(samplerate=sample_rate, startband=-bands, endband=bands, normfreq=center, desired_delay_sec=0.001)
        if numpy.max(self.frequencies) > (sample_rate / 2) * 0.9: raise ValueError('Max frequency too high')
        self.signal = None
        self.filter = LowPassFilter(1000, sample_rate, 2)
        self.gamma_output = None
        self.rectified = None
        self.compressed = None
        self.filtered = None
        self.result = None
        self.exponent = 0.4

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

    def run_model(self, signal):
        self.signal = signal
        self.gamma_output = self.run_gammatone()
        self.rectified = rectify(self.gamma_output)
        self.compressed = numpy.power(self.rectified, self.exponent)
        self.filtered = self.filter.run(self.compressed)
        self.result = rectify(self.filtered)
        self.result = numpy.mean(self.result, axis=0)
        return self.result

    def plot_filter(self):
        w, h = freqz(self.b, self.a)
        f = 0.5 * self.sample_rate * w / numpy.pi
        db = 20 * numpy.log10(abs(h))
        pyplot.plot(f, db)



if __name__ == "__main__":
    m = ModelWiegrebe(250000, 50000, 4)