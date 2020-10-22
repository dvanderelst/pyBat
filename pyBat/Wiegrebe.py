import numpy
from matplotlib import pyplot
from .Gammatone import GammatoneFilterbank
from scipy.fftpack import fft, ifft
from scipy.signal import freqz

from pyBat import Acoustics
from pyBat.Signal import LowPassFilter, rectify


def get_attenuation_coefficients(frequencies):
    attenuation_coefficients = []
    atmosphere = Acoustics.default_atmosphere()
    for frequency in frequencies:
        attenuation_coefficient = atmosphere.attenuation_coefficient(frequency) * -1
        attenuation_coefficients.append(attenuation_coefficient)
    return attenuation_coefficients


def attenuation_matrix(attenuation_coefficients, signal_length, sample_rate):
    atmosphere = Acoustics.default_atmosphere()
    duration = signal_length / sample_rate
    distance_max = atmosphere.soundspeed * duration / 2
    matrix = []
    for attenuation in attenuation_coefficients:
        scale = numpy.linspace(0, distance_max, signal_length) * attenuation *2
        matrix.append(scale)
    matrix = numpy.array(matrix)
    matrix = Acoustics.db2ratio(matrix)
    return matrix


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
        self.attenuation_matrix = None
        self.rectified = None
        self.compressed = None
        self.filtered = None
        self.matrix = None
        self.result = None
        self.exponent = 0.4
        # for frequency specific atmospheric filtering - if requested
        self.attenuation_coefficients = get_attenuation_coefficients(self.frequencies)

    @property
    def frequencies(self):
        return self.gammatone.centerfrequencies

    def set_lowpass(self, frequency):
        self.filter = LowPassFilter(frequency, self.sample_rate, 2)

    def run_gammatone(self):
        analyse = self.gammatone.analyze(self.signal)
        n = self.signal.size
        bm = numpy.zeros((0, n))
        for band in analyse:
            response = numpy.real(band[0])
            bm = numpy.vstack((bm, response))
        return bm

    def run_model(self, signal, apply_attenuation=False):
        self.signal = signal
        self.gamma_output = self.run_gammatone()
        if apply_attenuation:
            signal_length = self.signal.size
            matrix = attenuation_matrix(self.attenuation_coefficients,signal_length, self.sample_rate)
            self.gamma_output = self.gamma_output * matrix
            self.attenuation_matrix = matrix

        self.rectified = rectify(self.gamma_output)
        self.compressed = numpy.power(self.rectified, self.exponent)
        self.filtered = self.filter.run(self.compressed)
        self.matrix = rectify(self.filtered)
        self.result = numpy.mean(self.matrix, axis=0)
        return self.result

    def plot_filter(self):
        w, h = freqz(self.filter.b, self.filter.a)
        f = 0.5 * self.sample_rate * w / numpy.pi
        db = 20 * numpy.log10(abs(h))
        pyplot.plot(f, db)

    def plot_matrix(self):
        matrix = self.matrix
        y = self.frequencies
        max_time = (1 / self.sample_rate) * matrix.shape[1]
        x = numpy.linspace(0, 1, matrix.shape[1]) * max_time
        pyplot.contourf(x, y, matrix)
        pyplot.show()


if __name__ == "__main__":

    import Signal
    from matplotlib import pyplot
    emission = Signal.sweep(75000, 25000, 0.001, sample_rate=250000, method='hyperbolic', ramp=20)
    emission = emission[1]
    m = ModelWiegrebe(250000, 50000, 4, emission=emission)
    result = m.run_model(emission, dechirp=False, apply_attenuation=True)



    pyplot.plot(emission)
    pyplot.show()
    pyplot.plot(result)
    pyplot.show()

    pyplot.imshow(m.gamma_output)
    pyplot.show()
