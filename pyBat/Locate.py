import time
import numpy
from pyBat import Acoustics, Geometry
import random
from pyBat.Directivity import TransferFunction
from scipy.spatial.distance import cdist


class Locator:
    def __init__(self, source, freq1, freq2, **kwargs):
        self.freq_list = Acoustics.make_erb_cfs(freq1, freq2)
        self.mean_frequency = numpy.mean(numpy.array(self.freq_list))
        self.atmosphere = Acoustics.default_atmosphere()

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

        self.templates = None
        self.az_array = None
        self.el_array = None
        self.az_grid = None
        self.el_grid = None

        self.transfer_function = TransferFunction(source, self.freq_list, **kwargs)
        self.make_templates()

    @property
    def attenuation_coefficient(self):
        return self.atmosphere.attenuation_coefficient(self.mean_frequency) * -1

    def spreading(self, distance):
        distance = numpy.array(distance, dtype='f')
        return self.spreading_parameter * numpy.log10(distance / self.reference_distance)

    def attenuation(self, distance):
        distance = numpy.array(distance, dtype='f')
        return self.attenuation_coefficient * distance * 2

    def make_templates(self):
        eli = numpy.linspace(-90, 90, 181)
        azi = numpy.linspace(-90, 90, 181)
        extra = numpy.array([180])
        azi = numpy.concatenate((azi, extra))
        left, right = self.transfer_function.query(azi, eli)

        shape = left.shape
        template_rows = shape[0] * shape[1]
        template_cols = shape[2]

        left = numpy.reshape(left, (template_rows, template_cols))
        right = numpy.reshape(right, (template_rows, template_cols))
        azs, els = numpy.meshgrid(azi, eli)

        self.templates = numpy.concatenate((left, right), axis=1)
        self.az_array = numpy.reshape(azs, (-1, 1))
        self.el_array = numpy.reshape(els, (-1, 1))
        self.az_grid = azs
        self.el_grid = els

    def get_template_sph(self, real_az, real_el, left_only=False):
        template_index = numpy.argmin((self.az_array - real_az) ** 2 + (self.el_array - real_el) ** 2)
        template = self.templates[template_index, :]
        template = numpy.reshape(template, (1, -1))
        length = int(template.shape[1] / 2)
        if left_only: template = template[:, 0:length]
        return template

    def get_template_cart(self, real_x, real_y, real_z, left_only=False):
        real_az, real_el, _ = Geometry.cart2sph(real_x, real_y, real_z)
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


if __name__ == "__main__":
    import cProfile
    pr = cProfile.Profile()
    l = Locator('pd01', 30000, 50000)
    pr.enable()
    for x in range(0,100):l.locate(10, 10, 2)
    pr.disable()
    pr.print_stats(sort='time')