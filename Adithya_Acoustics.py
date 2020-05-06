from pyBat import Acoustics, Signal, Call
import numpy
from matplotlib import pyplot




class RobotSonar:
    def __init__(self):
        self.emission_duration = 0.0025
        self.emission_frequency = 41000
        self.fs = 250000



        emission_samples = int(self.fs * self.emission_duration)
        emission_time = numpy.linspace(0, self.emission_duration, emission_samples)
        emission = numpy.sin(2 * numpy.pi * self.emission_frequency * emission_time)
        # emission = chirp(emission_time, f0=50000, f1=25000, t1=emission_duration, method='quadratic')
        emission_window = Signal.signal_ramp(emission_samples, 10)
        self.emission = emission * emission_window
        self.caller = Call.Call('pd01', freq_list= [41000])


    def call(self, azimuths, elevation, distance):
        result = self.caller.call(azimuths, elevation, distance)
        delays = result['delays']
        echoes_left = result['echoes_left']
        echoes_right = result['echoes_right']

        impulse_response_left = Acoustics.make_impulse_response(delays, echoes_left, self.emission_duration, self.fs)
        impulse_response_right = Acoustics.make_impulse_response(delays, echoes_right, self.emission_duration, self.fs)

        left_ir = impulse_response_left['ir_result']
        right_ir = impulse_response_right['ir_result']

        echo_sequence_left = numpy.convolve(self.emission, left_ir, mode='same')
        echo_sequence_right = numpy.convolve(self.emission, right_ir, mode='same')

        result['left_echo_sequence'] = echo_sequence_left
        result['right_echo_sequence'] = echo_sequence_right
        return result


if __name__ == "__main__":
    c = RobotSonar()

    azimuths = numpy.linspace(-90,90,100)
    elevations = numpy.zeros(100)
    distances = numpy.ones(100)

    result = c.call(azimuths, elevations, distances)

    pyplot.plot(result['echoes_left'])
    pyplot.plot(result['echoes_right'])
    pyplot.show()

    azimuths = numpy.zeros(3)
    elevations = numpy.zeros(3)
    distances = numpy.linspace(2,3,3)


    result = c.call(azimuths, elevations, distances)
    pyplot.plot(result['left_echo_sequence'])
    pyplot.show()
    pyplot.plot(result['echoes_left'])
    pyplot.show()
