# positive error --> negative output
# negative error --> positive output

import numpy
from matplotlib import pyplot
from pyBat import Logger


class Controller:
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd
        self.logger = Logger.Logger()

    def add_sample(self, time, error):
        self.logger['time'] = time
        self.logger['error'] = error

    def output(self, time_stamps=None, verbose=False, simple_output=True):
        time, error = self.logger.get_history('error', time_stamps=time_stamps, return_time=True)
        gradient = self.logger.get_history('error', time_stamps=time_stamps, gradient=True)
        proportional = self.kp * error
        derivational = self.kd * gradient
        proportional[numpy.isnan(proportional)] = 0
        derivational[numpy.isnan(derivational)] = 0
        output = -(proportional + derivational)

        if verbose:
            print('>> T', numpy.round(time_stamps, 2))
            print('>> P', numpy.round(proportional, 2))
            print('>> D', numpy.round(derivational, 2))
            print('>> O', numpy.round(output, 2))
        if simple_output: return output
        results = {}
        results['error'] = error
        results['output'] = output
        results['proportional'] = proportional
        results['derivational'] = derivational
        results['time'] = time
        return results

    def plot(self, title=''):
        result = self.output(time_stamps='all', simple_output=False)
        time = result['time']
        error = result['error']
        output = result['output']
        pyplot.figure()
        pyplot.subplot(121)
        pyplot.plot(time, error)
        pyplot.xlabel('Time')
        pyplot.ylabel('Error')
        pyplot.grid()
        pyplot.title(title)
        pyplot.subplot(122)
        pyplot.plot(time, output)
        pyplot.xlabel('Time')
        pyplot.ylabel('Output')
        pyplot.grid()
        pyplot.tight_layout()


# a = Controller(1, 0.5)
# a.add_sample(0, 1)
# a.add_sample(0.1, 1.2)
# a.add_sample(0.3, 1.10)

# d = a.interpolate([0.15, 0.15])
# print(d)
# print(a.output(time_stamps=None, verbose=True))
