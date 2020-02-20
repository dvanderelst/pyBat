import numpy
import math
from matplotlib import pyplot
from pyBat import Misc, Logger, Geometry

numpy.set_printoptions(precision=3)


class Bat:
    def __init__(self):
        self.body_abs = Geometry.Frame()
        self.head_rel = Geometry.Frame()
        self.logger = Logger.Logger()
        self.clock = 0

    @property
    def head_abs(self):
        head_abs = Geometry.Frame(quaternion=self.body_abs.quaternion, position=self.body_abs.position)
        head_abs.apply_quaternion(self.head_rel.quaternion)
        return head_abs

    @property
    def x(self):
        return self.body_abs.x

    @property
    def y(self):
        return self.body_abs.y

    @property
    def z(self):
        return self.body_abs.z

    @property
    def position(self):
        return self.body_abs.position

    def motion(self, **kwargs):
        time = kwargs.pop('time', 0)
        nr_of_steps = math.ceil(time / 0.01)
        nr_of_steps = kwargs.pop('steps', nr_of_steps)
        speed = kwargs.pop('speed', 0)
        # Parameters are velocities
        body_yaw = kwargs.pop('body_y', 0)
        body_pitch = kwargs.pop('body_p', 0)
        body_roll = kwargs.pop('body_r', 0)

        head_yaw = kwargs.pop('head_y', 0)
        head_pitch = kwargs.pop('head_p', 0)
        head_roll = kwargs.pop('head_r', 0)

        keys = kwargs.keys()
        if len(keys) > 0: raise ValueError('Unused keyword arguments passed')

        #head_yaw = head_yaw - body_yaw
        #head_pitch = head_pitch - body_pitch
        #head_roll = head_roll - body_roll

        if nr_of_steps > 1:
            time_steps = numpy.linspace(0, time, nr_of_steps)
            time_steps = numpy.diff(time_steps)
        else:
            time_steps = [time]

        for step in time_steps:
            self.body_abs.move(time=step, yaw=body_yaw, pitch=body_pitch, roll=body_roll, speed=speed)
            self.head_rel.move(time=step, yaw=head_yaw, pitch=head_pitch, roll=head_roll, speed=0)

            self.head_rel.limit_rotations(yaw_limit=90, pitch_limit=45, roll_limit=60)
            self.body_abs.limit_rotations(pitch_limit=30, roll_limit=30)

            self.clock += step
            self.update_log()

    def bat2world(self, bat, frame='a'):
        if frame.startswith('b'): world = self.body_abs.frame2world(bat)
        if frame.startswith('seed_points'): world = self.head_rel.frame2world(bat)
        if frame.startswith('a'): world = self.head_abs.frame2world(bat)
        return world

    def world2bat(self, world, frame='a', spherical=False):
        if frame.startswith('b'): bat = self.body_abs.world2frame(world, spherical=spherical)
        if frame.startswith('seed_points'): bat = self.head_rel.world2frame(world, spherical=spherical)
        if frame.startswith('a'): bat = self.head_abs.world2frame(world, spherical=spherical)
        return bat

    def update_log(self):
        body = self.body_abs.spherical_angles
        head = self.head_rel.spherical_angles
        total = self.head_abs.spherical_angles
        # Add time data
        self.logger['time'] = self.clock
        # Add position data
        self.logger['x', 'y', 'z'] = [self.x, self.y, self.z]
        # Add body data
        self.logger['a_b', 'e_b'] = body
        # Add head data
        self.logger['a_h', 'e_h'] = head
        # Add total data
        self.logger['a_t', 'e_t'] = total
        # Add body quaternion
        self.logger['b0', 'b1', 'b2', 'b3'] = self.body_abs.quaternion
        # Add head quaternion
        self.logger['r0', 'r1', 'r2', 'r3'] = self.head_rel.quaternion
        # Add total quaternion
        self.logger['a0', 'a1', 'a2', 'a3'] = self.head_abs.quaternion

    def get_history(self, field, time_stamps=None, gradient=False, unwrap=True, absolute=False):
        apply_unwrap = False
        angle_fields = ['a_b', 'e_b', 'a_h', 'e_h', 'a_t', 'e_t']
        if field in angle_fields and unwrap: apply_unwrap = True
        data = self.logger.get_history(field, time_stamps, gradient, apply_unwrap, False, absolute)
        return data

    def get_history_vectors(self, time_stamps=None, frame='t'):
        if frame.startswith('b'): labels = ['b0', 'b1', 'b2', 'b3']
        if frame.startswith('seed_points'): labels = ['r0', 'r1', 'r2', 'r3']
        if frame.startswith('a'): labels = ['a0', 'a1', 'a2', 'a3']
        q0 = self.get_history(labels[0], time_stamps=time_stamps)
        q1 = self.get_history(labels[1], time_stamps=time_stamps)
        q2 = self.get_history(labels[2], time_stamps=time_stamps)
        q3 = self.get_history(labels[3], time_stamps=time_stamps)

        x = self.get_history('x', time_stamps=time_stamps)
        y = self.get_history('y', time_stamps=time_stamps)
        z = self.get_history('z', time_stamps=time_stamps)

        quaternions = numpy.row_stack((q0, q1, q2, q3))
        positions = numpy.row_stack((x, y, z))

        quaternions = numpy.transpose(quaternions)
        positions = numpy.transpose(positions)

        quaternions = numpy.asmatrix(quaternions)
        positions = numpy.asmatrix(positions)
        vectors = numpy.zeros(shape=positions.shape)
        for index in range(0, len(time_stamps)):
            quaternion = Misc.mat2array(quaternions[index, :])
            position = Misc.mat2array(positions[index, :])
            frame = Geometry.Frame(quaternion=quaternion, position=position)
            vector = frame.direction_vector
            vector = Misc.normalize_vector(vector)
            vectors[index, :] = vector

        quaternions = Misc.mat2array(quaternions)
        positions = Misc.mat2array(positions)
        vectors = Misc.mat2array(vectors)
        return vectors, positions, quaternions

    def plot_path(self):
        x = self.logger['x']
        y = self.logger['y']
        z = self.logger['z']
        path_figure = pyplot.figure()
        axis = path_figure.gca(projection='3d')
        axis.plot(x, y, z, marker='o')
        axis.set_xlabel('X Label')
        axis.set_ylabel('Y Label')
        axis.set_zlabel('Z Label')

    def plot_quiver(self, time_stamps=None, view='top', length=0.05):
        t = self.logger['time']
        mn = t.min()
        mx = t.max()
        if time_stamps is None: time_stamps = self.logger['time']
        if type(time_stamps) == float: time_stamps = numpy.arange(mn, mx, time_stamps)
        vectors_head_abs, start_head_abs, _ = self.get_history_vectors(time_stamps=time_stamps, frame='a')
        vectors_body_abs, start_body_abs, _ = self.get_history_vectors(time_stamps=time_stamps, frame='b')

        vectors_head_abs *= length
        vectors_body_abs *= length

        max_body = numpy.max(numpy.abs(start_body_abs))
        max_total = numpy.max(numpy.abs(start_head_abs))
        max_value = max(max_body, max_total) + 0.5

        number = vectors_head_abs.shape[0]

        if view.startswith('3'):
            fig = pyplot.figure()
            ax = fig.gca(projection='3d')

        for index in range(0, number):
            t_x0 = start_head_abs[index, 0]
            t_y0 = start_head_abs[index, 1]
            t_z0 = start_head_abs[index, 2]
            t_x1 = vectors_head_abs[index, 0]
            t_y1 = vectors_head_abs[index, 1]
            t_z1 = vectors_head_abs[index, 2]

            b_x0 = start_body_abs[index, 0]
            b_y0 = start_body_abs[index, 1]
            b_z0 = start_body_abs[index, 2]
            b_x1 = vectors_body_abs[index, 0]
            b_y1 = vectors_body_abs[index, 1]
            b_z1 = vectors_body_abs[index, 2]

            if view.startswith('t'):
                a = pyplot.arrow(b_x0, b_y0, b_x1, b_y1, head_width=0.01, head_length=0.025, fc='k', ec='k', alpha=0.75)
                b = pyplot.arrow(t_x0, t_y0, t_x1, t_y1, head_width=0.01, head_length=0.025, fc='r', ec='r', alpha=0.50)
                pyplot.xlabel('x')
                pyplot.ylabel('y')


            if view.startswith('s'):
                a = pyplot.arrow(b_x0, b_z0, b_x1, b_z1, head_width=0.01, head_length=0.025, fc='k', ec='k', alpha=0.75)
                b = pyplot.arrow(t_x0, t_z0, t_x1, t_z1, head_width=0.01, head_length=0.025, fc='r', ec='r', alpha=0.50)
                pyplot.xlabel('x')
                pyplot.ylabel('z')

            if view.startswith('3'):
                ax.quiver(b_x0, b_y0, b_z0, b_x1, b_y1, b_z1, color='k', alpha=0.75)
                ax.quiver(t_x0, t_y0, t_z0, t_x1, t_y1, t_z1, color='r', alpha=0.75)

        if view.startswith('3'):
            pyplot.xlabel('x')
            pyplot.ylabel('y')
            return

        pyplot.xlim(-max_value, max_value)
        pyplot.ylim(-max_value, max_value)
        ax = pyplot.gca()
        ax.set_aspect(1)
        results = {}
        results['vectors_body'] = vectors_body_abs
        results['vectors_head'] = vectors_head_abs
        results['start_head'] = start_head_abs
        results['start_body'] = start_body_abs
        return results, a, b


# a = numpy.array([0, 0, 1])
# b = Bat()
#
# pitches = []
# yaws = []
# rolls = []
# for x in range(0, 10):
#     b.motion(head_pitch=45, speed=1, time=0.1)
#     b.head_rel.limit_rotations(pitch_limit=33)
#     rolls.append(b.head_abs.euler[0])
#     pitches.append(b.head_abs.euler[1])
#     yaws.append(b.head_abs.euler[2])
#
#
# b.plot_quiver(view='s')
# #pyplot.figure()
# #b.plot_quiver(view='t')
# print(b.body_abs.spherical_angles)
# print(b.head_rel.spherical_angles)
# print(b.head_abs.spherical_angles)
#
# print(b.head_abs.euler)



# Rotate.test_frame()
