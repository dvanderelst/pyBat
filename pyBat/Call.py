import time

import numpy

from pyBat import Acoustics, Directivity


def out_of_fov(azimuths, elevations):
    abs_azimuths = numpy.abs(azimuths)
    abs_elevations = numpy.abs(elevations)
    not_in_fov = 1 - ((abs_azimuths < 90) * (abs_elevations < 90))
    not_in_fov = not_in_fov == 1
    return not_in_fov


class Call:
    def __init__(self, source, freq_list, **kwargs):

        if source is None:
            self.transfer = None
        else:
            self.transfer = Directivity.TransferFunction(source=source, freq_list=freq_list, collapse=True, **kwargs)

        self.mean_frequency = numpy.mean(numpy.array(freq_list))
        self.atmosphere = Acoustics.default_atmosphere()
        self.attenuation_coefficient = self.atmosphere.attenuation_coefficient(self.mean_frequency) * -1

        # Parameters for 1 m ref value as given by Stilz
        self.reflection_parameter = -40
        self.spreading_parameter = -40
        self.call_level = 110
        self.detection_threshold = 20

    def call(self, azimuths, elevations, distances, noise=None, gains=None):
        start = time.time()
        azimuths = numpy.array(azimuths)
        elevations = numpy.array(elevations)
        distances = numpy.array(distances)

        outside_of_fov = out_of_fov(azimuths, elevations)

        spreading = self.spreading_parameter * numpy.log10(distances)
        attenuation = self.attenuation_coefficient * distances * 2
        echo_db = self.call_level + spreading + attenuation + self.reflection_parameter

        if noise is not None: echo_db = echo_db + noise
        if gains is not None: echo_db = echo_db + gains

        directivity = [0, 0]
        if self.transfer is not None: directivity = self.transfer.query(azimuths, elevations)
        echo_db_left = echo_db + directivity[0]
        echo_db_right = echo_db + directivity[1]

        echo_db_left[outside_of_fov] = 0
        echo_db_right[outside_of_fov] = 0

        echo_db_left[echo_db_left > self.call_level] = self.call_level
        echo_db_right[echo_db_right > self.call_level] = self.call_level
        max_echo = numpy.maximum(echo_db_left, echo_db_right)
        detection_left = echo_db_left > self.detection_threshold
        detection_right = echo_db_right > self.detection_threshold
        detection = numpy.maximum(detection_left, detection_right)
        delays = Acoustics.dist2delay(distances)
        end = time.time()
        duration = end - start
        result = {}
        result['echoes_left'] = echo_db_left
        result['echoes_right'] = echo_db_right
        result['detection_left'] = detection_left
        result['detection_right'] = detection_right
        result['detectable'] = detection
        result['delays'] = delays
        result['max'] = max_echo
        result['calculation_duration'] = duration
        result['directivity'] = directivity
        result['azimuth'] = azimuths
        result['elevation'] = elevations
        result['distances'] = distances
        return result


def process4avoidance(result, delta_t=0.1):
    detectable = result['detectable']
    delays = result['delays']
    detectable_delays = delays[detectable]
    if len(detectable_delays) == 0: min_delay = 9999
    if len(detectable_delays) > 0: min_delay = numpy.min(detectable_delays)
    max_delay = min_delay + delta_t
    selected = delays <= max_delay
    left = result['detection_left'] * selected
    right = result['detection_right'] * selected
    left_echoes = result['echoes_left'][left]
    right_echoes = result['echoes_right'][right]
    left_sum = Acoustics.incoherent_sum(left_echoes)
    right_sum = Acoustics.incoherent_sum(right_echoes)
    return left_sum, right_sum, min_delay


# if __name__ == "__main__":
#     source = 'pd01'
#     freq_list = [25000, 35000]
#
#     azs = [-30, 0, 30]
#     els = [0, 0, 0]
#     distances = [3, 3, 3]
#
#     c = Call(source=source, freq_list=freq_list, pitch=30)
#     print('call1')
#     result = c.call(azs, els, distances)
#     print('call2')
#     result = c.call(azs, els, distances)
