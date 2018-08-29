import Directivity
import Acoustics
import numpy
import time


class Call:
    def __init__(self, source, freq_list, **kwargs):
        self.mean_frequency = numpy.mean(numpy.array(freq_list))
        self.transfer = Directivity.TransferFunction(source=source, freq_list=freq_list, collapse=True, **kwargs)
        self.atmosphere = Acoustics.default_atmosphere()
        self.attenuation_coefficient = self.atmosphere.attenuation_coefficient(self.mean_frequency) * -1
        self.reference_distance = 0.1
        self.reflection_parameter = -20  # Parameter C1 in Stilz and Schnitzler but for 10 cm
        self.spreading_parameter = -40  # Parameter C2 in Stilz and Schnitzler
        self.call_level = 120
        self.detection_threshold = 20

    def call(self, azimuths, elevations, distances, noise=None, filter_distance=False):
        start = time.time()
        azimuths = numpy.array(azimuths)
        elevations = numpy.array(elevations)
        distances = numpy.array(distances)

        if filter_distance:
            # based on distance
            azimuths = azimuths[distances < filter_distance]
            elevations = elevations[distances < filter_distance]
            noise = noise[distances < filter_distance]
            distances = distances[distances < filter_distance]
            # based on angle
            #abs_az = numpy.abs(azimuths)
            #in_view = abs_az < 120
            #azimuths = azimuths[in_view]
            #elevations = elevations[in_view]
            #noise = noise[in_view]
            #distances = distances[in_view]

        spreading = self.spreading_parameter * numpy.log10(distances / self.reference_distance)
        attenuation = self.attenuation_coefficient * distances * 2
        echo_db = self.call_level + spreading + attenuation + self.reflection_parameter
        if noise is not None: echo_db = echo_db + noise

        directivity = self.transfer.query(azimuths, elevations)
        echo_db_left = echo_db + directivity[0]
        echo_db_right = echo_db + directivity[1]
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
        result['duration'] = duration
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


if __name__ == "__main__":
    source = 'pd01'
    freq_list = [25000, 35000]

    azs = [-30, 0, 30]
    els = [0, 0, 0]
    distances = [3, 3, 3]

    c = Call(source=source, freq_list=freq_list, pitch=30)
    print('call1')
    result = c.call(azs, els, distances)
    print('call2')
    result = c.call(azs, els, distances)
