import numpy
import random
from pyBat import Misc
from pyBat import Logger
from pyBat import Frame


class Insect:
    def __init__(self, start_azimuth, start_elevation, start_distance):
        x, y, z = Frame.sph2cart(start_azimuth, start_elevation, start_distance)
        self.frame = Frame.Frame(position=numpy.array([x, y, z]))
        self.yaw_speed = 0
        self.pitch_speed = 0
        self.speed = 0
        self.clock = 0
        self.section_remainder = 0
        self.time_scale = 1
        self.logger = Logger.Logger()

        # random start orientation
        yaw_rotation = random.randrange(-180, 180)
        pitch_rotation = random.randrange(-45, 45) * 0
        self.frame.move(time=1, yaw=yaw_rotation, pitch=pitch_rotation)

    @property
    def position(self):
        return self.frame.position

    @property
    def x(self):
        return self.frame.position[0]

    @property
    def y(self):
        return self.frame.position[1]

    @property
    def z(self):
        return self.frame.position[2]

    def update_motion_parameters(self):
        self.yaw_speed = numpy.random.uniform(low=-180, high=180)
        self.pitch_speed = numpy.random.uniform(low=-90, high=90)
        #todo randrange 0 - 2?
        self.speed = numpy.random.uniform(0.5, 2)
        #self.speed = 0.1 * numpy.random.randn() + 1
        self.section_remainder = numpy.random.exponential(scale=self.time_scale)

    def motion(self, time_step):
        if self.section_remainder <= 0: self.update_motion_parameters()
        self.frame.move(time=time_step, yaw=self.yaw_speed, pitch=self.pitch_speed, speed=self.speed)
        self.clock = self.clock + time_step
        self.section_remainder = self.section_remainder - time_step
        if abs(self.z) > 0.5: self.frame.position[2] = 0.5 * Misc.sign(self.z)
        self.logger['time'] = self.clock
        self.logger['x', 'y', 'z'] = [self.x, self.y, self.z]
        self.logger['speed'] = self.speed

    def run(self, duration=1, time_step=0.0025):
        while self.clock < duration: self.motion(time_step=time_step)

    def get_position(self, time_stamps):
        x = self.logger.get_history('x', time_stamps=time_stamps)
        y = self.logger.get_history('y', time_stamps=time_stamps)
        z = self.logger.get_history('z', time_stamps=time_stamps)
        position = numpy.column_stack((x, y, z))
        return position

    def get_speed(self, time_stamps):
        speed = self.logger.get_history('speed', time_stamps=time_stamps)
        return speed


if __name__ == "__main__":
    from matplotlib import pyplot

    i = Insect(0, 0, 0)
    i.run(15)

    p = i.get_position('all')
    t = i.logger.time

    v = numpy.gradient(p, t, axis=0)
    n = numpy.linalg.norm(v, axis=1)

    i.logger.plot(['x', 'y', 'z', 'speed'])
    pyplot.plot(t, n)
    x = i.logger['x']
    y = i.logger['y']

    pyplot.figure()
    pyplot.plot(x, y)
    pyplot.title('top view')



    pyplot.figure()

