import os
import pandas
import numpy
from matplotlib import pyplot
from pyBat import Misc


class Jones:
    def __init__(self):
        script_path = os.path.realpath(__file__)
        script_path, b = os.path.split(script_path)
        data_path = os.path.join(script_path, 'data', 'Jones.xls')
        self.data = pandas.read_excel(data_path, sheetname='Jones')
        self.fitted_function = None
        self.fit_function()

    def fit_function(self):
        speed_data = numpy.log10(self.data.Speed)
        angular_data = numpy.log10(self.data.Angular)
        z = numpy.polyfit(speed_data, angular_data, 1)
        self.fitted_function = numpy.poly1d(z)

    def get_angular_velocity(self, speed, log=False):
        if not Misc.iterable(speed): speed = [speed]
        if speed is None: speed = numpy.sort(self.data.Speed)
        speed = numpy.array(speed, dtype='f')
        # Handle speeds outside the fitted limits
        speed_data = self.data.Speed
        mn = numpy.min(speed_data)
        mx = numpy.max(speed_data)

        # Fix angular velocity to min/max outside of reported range
        speed[speed < mn] = mn
        speed[speed > mx] = mx

        speed = numpy.log10(speed)
        # Apply the fitted function
        angular = self.fitted_function(speed)
        if log: return angular
        angular = 10 ** angular
        return angular

    def plot_data(self):
        speed_data = self.data.Speed
        angular_data = self.data.Angular
        mn = numpy.min(speed_data)
        mx = numpy.max(speed_data)
        speed_i1 = numpy.linspace(mn, mx, 100)
        fitted_i1 = self.get_angular_velocity(speed_i1)
        speed_i2 = numpy.linspace(0, 4, 100)
        fitted_i2 = self.get_angular_velocity(speed_i2)
        a = pyplot.scatter(speed_data, angular_data)
        b, = pyplot.plot(speed_i1, fitted_i1, '-', color='red')
        c, = pyplot.plot(speed_i2, fitted_i2, '--', color='red')
        pyplot.xlabel('Bat Velocity m/s')
        pyplot.ylabel('Angular Velocity (deg./s)')
        pyplot.show()
        return a, b, c


#j = Jones()
#a = j.get_angular_velocity(1)
#b = j.get_angular_velocity(2)
#c = j.get_angular_velocity([1,2])
#print(a, b, c)
#j.plot_data()
