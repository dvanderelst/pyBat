@property
def rotation_matrix_total(self):
    return self.rotation_matrix_body * self.rotation_matrix_head


@property
def body_angles(self):
    azimuth_body, elevation_body = Rotate.euler_angles(self.rotation_matrix_body)
    return azimuth_body, elevation_body


@property
def head_angles(self):
    azimuth_head, elevation_head = Rotate.euler_angles(self.rotation_matrix_head)
    return azimuth_head, elevation_head


@property
def total_angles(self):
    azimuth_total, elevation_total = Rotate.euler_angles(self.rotation_matrix_total)
    return azimuth_total, elevation_total


@property
def x(self):
    return float(self.position[0, 0])


@property
def y(self):
    return float(self.position[0, 1])


@property
def z(self):
    return float(self.position[0, 2])


def apply_angle_boundaries(self, az_max, el_max, frame):
    if not self.bounded: return
    if frame.startswith('h'):
        azimuth = self.head_angles[0]
        elevation = self.head_angles[1]
    if frame.startswith('b'):
        azimuth = self.body_angles[0]
        elevation = self.body_angles[1]
    if abs(azimuth) > az_max: azimuth = az_max * numpy.sign(azimuth)
    if abs(elevation) > el_max: elevation = el_max * numpy.sign(elevation)
    new_matrix = Rotate.rotation_matrix_frame2world(azimuth, elevation)
    start = numpy.matrix(numpy.eye(3, 3))
    rotation_matrix = start * new_matrix
    if frame.startswith('h'): self.rotation_matrix_head = rotation_matrix
    if frame.startswith('b'): self.rotation_matrix_body = rotation_matrix


def move_bat_step(self, azimuth, elevation, distance):
    vector = numpy.matrix([0, 0, distance])
    vector = numpy.transpose(vector)
    matrix = Rotate.rotation_matrix_frame2world(azimuth, elevation)
    displacement = self.rotation_matrix_body * matrix * vector
    self.position += numpy.transpose(displacement)
    self.rotation_matrix_body = self.rotation_matrix_body * matrix
    self.apply_angle_boundaries(1000, 30, frame='b')


def turn_head_step(self, azimuth, elevation):
    matrix = Rotate.rotation_matrix_frame2world(azimuth, elevation)
    self.rotation_matrix_head = self.rotation_matrix_head * matrix
    self.apply_angle_boundaries(90, 45, frame='h')


def reset_head(self):
    self.rotation_matrix_head = numpy.matrix(numpy.eye(3, 3))


def turn_towards(self, x=None, y=None, z=None):
    self.reset_head()
    if x is None:
        self.move_bat_step(180, 0, 0)
    else:
        new_x, new_y, new_z = self.world2bat_cart(x, y, z)
        az, el, dist = Rotate.cart2sph(new_x, new_y, new_z)
        self.move_bat_step(az, el, 0)


def look_at(self, x, y, z):
    new_x, new_y, new_z = self.world2bat_cart(x, y, z)
    az, el, dist = Rotate.cart2sph(new_x, new_y, new_z)
    self.turn_head_step(az, el)


def motion(self, body_az=0, body_el=0, speed=0, head_az=0, head_el=0, time=0, steps=1, update_log=True):
    delta_time = time / steps
    delta_body_az = body_az * delta_time
    delta_body_el = body_el * delta_time
    delta_head_az = head_az * delta_time
    delta_head_el = head_el * delta_time
    delta_distance = speed * delta_time
    for index in range(0, steps):
        self.turn_head_step(delta_head_az, delta_head_el)
        self.move_bat_step(delta_body_az, delta_body_el, delta_distance)
        self.clock += delta_time
        if self.auto_update or update_log: self.update_log()


def world2bat(self, world, frame='h'):
    world = numpy.reshape(world, (-1, 3))
    if frame.startswith('b'): matrix = self.rotation_matrix_body
    if frame.startswith('h'): matrix = self.rotation_matrix_total
    matrix_invert = numpy.linalg.inv(matrix)
    world = world.astype(float)
    world[:, 0] = world[:, 0] - self.position[0, 0]
    world[:, 1] = world[:, 1] - self.position[0, 1]
    world[:, 2] = world[:, 2] - self.position[0, 2]
    new = matrix_invert * numpy.transpose(world)
    new = numpy.transpose(new)
    return new


def world2bat_cart(self, x, y, z, frame='h'):
    matrix = numpy.column_stack((x, y, z))
    new = self.world2bat(matrix, frame)
    new_x = mat2array(new[:, 0])
    new_y = mat2array(new[:, 1])
    new_z = mat2array(new[:, 2])
    return new_x, new_y, new_z


def world2bat_sph(self, az, el, dist, frame='h'):
    x, y, z = Rotate.sph2cart(az, el, dist)
    new_x, new_y, new_z = self.world2bat_cart(x, y, z, frame)
    new_az, new_el, new_dist = Rotate.cart2sph(new_x, new_y, new_z)
    return new_az, new_el, new_dist


def call_cart(self, x, y, z, sonar=None):
    if sonar is None: sonar = self.sonar
    new_x, new_y, new_z = self.world2bat_cart(x, y, z)
    left, right, delay = sonar.call_cart(new_x, new_y, new_z)
    return left, right, delay


def call_sph(self, azimuth, elevation, distance, sonar=None):
    if sonar is None: sonar = self.sonar
    x, y, z = Rotate.sph2cart(azimuth, elevation, distance)
    return self.call_cart(x, y, z, sonar)


def update_log(self):
    body = self.body_angles
    head = self.head_angles
    total = self.total_angles
    # Add time data
    self.logger['time'] = self.clock
    # Add position data
    self.logger['x','y','z'] = [self.x, self.y, self.z]
    # Add body data
    self.logger['a_b', 'e_b'] = body
    # Add head data
    self.logger['a_h', 'e_h'] = head
    # Add total data
    self.logger['a_t', 'e_t'] = total



def get_history(self, field, time_stamps=None, gradient=False, unwrap=True, absolute=False):
    angle_fields = ['a_b', 'e_b', 'a_h', 'e_h', 'a_t', 'e_t']
    apply_unwrap = False
    if field in angle_fields and unwrap: apply_unwrap = True
    data = self.logger.get_history(field, time_stamps, gradient, apply_unwrap, False, absolute)
    return data


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


def get_history_rotation(self, time_stamp, frame='h'):
    if frame.startswith('h'):
        azimuth = self.get_history('a_t', time_stamp)
        elevation = self.get_history('e_t', time_stamp)
    if frame.startswith('b'):
        azimuth = self.get_history('a_b', time_stamp)
        elevation = self.get_history('e_b', time_stamp)
    matrix = Rotate.rotation_matrix_frame2world(azimuth, elevation)
    return matrix


def get_history_vector(self, time_stamp, frame='h'):
    vector = Misc.unit_vector(1)
    matrix = self.get_history_rotation(time_stamp, frame)
    pointing = matrix * numpy.matrix(vector)
    pointing = Rotate.mat2array(pointing)
    return pointing


def get_history_vectors(self, time_stamps, frame='h'):
    locations = numpy.zeros((0, 3))
    vectors = numpy.zeros((0, 3))
    for time_stamp in time_stamps:
        x = self.get_history('x', time_stamp)
        y = self.get_history('y', time_stamp)
        z = self.get_history('z', time_stamp)
        location = numpy.array([float(x), float(y), float(z)], dtype='f')
        vector = self.get_history_vector(time_stamp, frame)
        vectors = numpy.vstack((vectors, vector))
        locations = numpy.vstack((locations, location))
    return vectors, locations


def plot_quiver(self, time_stamps=None, view='top', length=0.05):
    if time_stamps is None: time_stamps = self.logger['time']
    vectors_head, start_head = self.get_history_vectors(time_stamps, 'h')
    vectors_body, start_body = self.get_history_vectors(time_stamps, 'b')

    vectors_head *= length
    vectors_body *= length

    max_body = numpy.max(numpy.abs(start_body))
    max_head = numpy.max(numpy.abs(start_head))
    max_value = max(max_body, max_head) + 0.5

    number = vectors_head.shape[0]

    if view.startswith('3'):
        fig = pyplot.figure()
        ax = fig.gca(projection='3d')

    for index in range(0, number):
        h_x0 = start_head[index, 0]
        h_y0 = start_head[index, 1]
        h_z0 = start_head[index, 2]
        h_x1 = vectors_head[index, 0]
        h_y1 = vectors_head[index, 1]
        h_z1 = vectors_head[index, 2]

        b_x0 = start_body[index, 0]
        b_y0 = start_body[index, 1]
        b_z0 = start_body[index, 2]
        b_x1 = vectors_body[index, 0]
        b_y1 = vectors_body[index, 1]
        b_z1 = vectors_body[index, 2]

        if view.startswith('t'):
            a = pyplot.arrow(b_z0, b_x0, b_z1, b_x1, head_width=0.01, head_length=0.025, fc='k', ec='k', alpha=0.75)
            b = pyplot.arrow(h_z0, h_x0, h_z1, h_x1, head_width=0.01, head_length=0.025, fc='r', ec='r', alpha=0.50)

        if view.startswith('s'):
            a = pyplot.arrow(b_z0, b_y0, b_z1, b_y1, head_width=0.01, head_length=0.025, fc='k', ec='k', alpha=0.75)
            b = pyplot.arrow(h_z0, h_y0, h_z1, h_y1, head_width=0.01, head_length=0.025, fc='r', ec='r', alpha=0.50)

        if view.startswith('3'):
            ax.quiver(b_z0, b_x0, b_y0, b_z1, b_x1, b_y1, color='k', alpha=0.75)
            ax.quiver(h_z0, h_x0, h_y0, h_z1, h_x1, h_y1, color='r', alpha=0.75)

    if view.startswith('3'):
        pyplot.xlabel('z')
        pyplot.ylabel('x')
        return

    pyplot.xlim(-max_value, max_value)
    pyplot.ylim(-max_value, max_value)
    ax = pyplot.gca()
    ax.set_aspect(1)
    ax.invert_xaxis()
    results = {}
    results['vectors_body'] = vectors_body
    results['vectors_head'] = vectors_head
    results['start_head'] = start_head
    results['start_body'] = start_body
    return results, a, b


def print_orientation(self):
    print('Position:', self.position)
    print('Body: ', self.body_angles)
    print('Head: ', self.head_angles)
    print('Total: ', self.total_angles)
