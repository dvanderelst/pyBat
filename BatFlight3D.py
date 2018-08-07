from matplotlib import pyplot
from scipy.interpolate import interp1d
from pyBat import Bat
from pyBat import Misc
from pyBat import Smoothn
from pyBat import Logger
from pyBat import Insect
from pyBat import Jones
from pyBat import Hearing
import numpy
import math


def bat_flight(parameters, my_bug):
    mx_spd = parameters['mx_spd']
    cpl_stn = parameters['cp_stn']
    fx_hed = parameters['fx_hed']
    mx_acc = parameters['mx_acc']
    fx_spd = parameters['fx_spd']

    max_time = 15
    capture_distance = 0.1

    # Set up bat
    my_bat = Bat.Bat()

    # set up locator
    locator = Hearing.Locator()

    # set up logger
    logger = Logger.Logger()

    # Set up profiles
    jones_curvature_profile = Jones.Jones()
    ipi_profile = get_ipi_profile()


    # Initialize
    previous_head_az_rotation_speed = 0
    previous_head_el_rotation_speed = 0

    previous_bat_x = 0
    previous_bat_y = 0
    previous_bat_z = 0

    bug_position = my_bug.get_position(time_stamps=0)
    previous_bug_x = bug_position[0][0]
    previous_bug_y = bug_position[0][1]
    previous_bug_z = bug_position[0][2]

    my_bat.update_log()
    time_step = ipi_profile(1000)
    v_bat = mx_spd / 2

    for step in range(0, 5000):
        # Locate insect
        bug_position = my_bug.get_position(time_stamps=my_bat.clock)
        bug_position = bug_position[0]
        target_az, target_el, target_dist = my_bat.world2bat(bug_position, frame='a', spherical=True)
        perceived_location = locator.locate(target_az, target_el, target_dist)

        target_az_perceived = perceived_location['az']
        target_el_perceived = perceived_location['el']
        out_of_focus = perceived_location['out_of_fov']

        # check break conditions
        bat_x_int = numpy.linspace(previous_bat_x, my_bat.x, 100)
        bat_y_int = numpy.linspace(previous_bat_y, my_bat.y, 100)
        bat_z_int = numpy.linspace(previous_bat_z, my_bat.z, 100)

        bug_x_int = numpy.linspace(previous_bug_x, bug_position[0], 100)
        bug_y_int = numpy.linspace(previous_bug_y, bug_position[1], 100)
        bug_z_int = numpy.linspace(previous_bug_z, bug_position[2], 100)

        D = numpy.sqrt((bat_x_int - bug_x_int) ** 2 + (bat_y_int - bug_y_int) ** 2 + (bat_z_int - bug_z_int) ** 2)
        interpolated_distance = numpy.min(D)

        if interpolated_distance < capture_distance: break
        if target_dist > 5: break
        if my_bat.clock > max_time: break
        if out_of_focus: break

        # get head orientation
        head_az_position, head_el_position = my_bat.head_rel.spherical_angles

        # Get the head rotation elevation
        # if abs(target_az_perceived) < 3: target_az_perceived = 0
        # head_az_rotation_speed = float(logger.get_history('target_az_perceived', time_stamps=-0.05) / time_step)
        head_az_rotation_speed = float(target_az_perceived / time_step)
        # if time_step <= 0.005: head_az_rotation_speed = previous_head_az_rotation_speed
        acceleration_head_az = (head_az_rotation_speed - previous_head_az_rotation_speed) / time_step
        if abs(acceleration_head_az) > mx_acc: acceleration_head_az = Misc.sign(acceleration_head_az) * mx_acc
        head_az_rotation_speed = previous_head_az_rotation_speed + acceleration_head_az * time_step
        if abs(head_az_rotation_speed) > 400: head_az_rotation_speed = 400 * Misc.sign(head_az_rotation_speed)

        # Get the head rotation elevation
        # if abs(target_el_perceived) < 3: target_el_perceived = 0
        # head_el_rotation_speed = float(logger.get_history('target_el_perceived', time_stamps=-0.05) / time_step)
        head_el_rotation_speed = float(target_el_perceived / time_step)
        # if time_step <=0.005: head_el_rotation_speed = previous_head_el_rotation_speed
        acceleration_head_el = (head_el_rotation_speed - previous_head_el_rotation_speed) / time_step
        if abs(acceleration_head_el) > mx_acc: acceleration_head_el = Misc.sign(acceleration_head_el) * mx_acc
        head_el_rotation_speed = previous_head_el_rotation_speed + acceleration_head_el * time_step
        if abs(head_el_rotation_speed) > 400: head_el_rotation_speed = 400 * Misc.sign(head_el_rotation_speed)

        previous_head_el_rotation_speed = head_el_rotation_speed
        previous_head_az_rotation_speed = head_az_rotation_speed

        power = 2
        # Get optimal speed
        v_optimal_az = math.cos(math.radians(head_az_position * 1)) ** power * mx_spd
        v_optimal_el = math.cos(math.radians(head_el_position * 2)) ** power * mx_spd

        # Get optimal speed for head locked
        if fx_hed:
            v_optimal_az = math.cos(math.radians(target_az_perceived * 1)) ** power * mx_spd
            v_optimal_el = math.cos(math.radians(target_el_perceived * 2)) ** power * mx_spd


        # to avoid complex numbers
        v_optimal_az = abs(v_optimal_az)
        v_optimal_el = abs(v_optimal_el)
        v_optimal = numpy.min((v_optimal_az, v_optimal_el))

        # Constraints on speed
        acceleration = float((v_optimal - v_bat) / time_step)
        max_acceleration = 5
        min_accelaration = -10
        if acceleration > max_acceleration: acceleration = max_acceleration
        if acceleration < min_accelaration: acceleration = min_accelaration
        v_bat = v_bat + (acceleration * time_step)
        if v_bat < 0.5: v_bat = 0.5
        if v_bat > mx_spd: v_bat = mx_spd
        if fx_spd: v_bat = mx_spd / 2

        # Get gain and delay
        coupling_gn, coupling_dl = get_gain_setting(cpl_stn)
        gain = coupling_gn(target_dist)
        time_constant = coupling_dl(target_dist)

        if not fx_hed:
            body_az_rot_speed = gain * float(my_bat.get_history('a_h', time_stamps=-time_constant))
            body_el_rot_speed = gain * float(my_bat.get_history('e_h', time_stamps=-time_constant))
        else:
            body_az_rot_speed = gain * float(logger.get_history('target_az_perceived', time_stamps=-time_constant))
            body_el_rot_speed = gain * float(logger.get_history('target_el_perceived', time_stamps=-time_constant))

        max_b_rotation = jones_curvature_profile.get_angular_velocity(v_bat)
        if abs(body_el_rot_speed) > max_b_rotation: body_el_rot_speed = max_b_rotation * Misc.sign(body_el_rot_speed)
        if abs(body_az_rot_speed) > max_b_rotation: body_az_rot_speed = max_b_rotation * Misc.sign(body_az_rot_speed)

        h_pt_v = float(- head_el_rotation_speed)
        h_yw_v = float(+ head_az_rotation_speed)

        b_pt_v = float(- body_el_rot_speed)
        b_yw_v = float(+ body_az_rot_speed)

        if fx_hed: h_pt_v = 0;h_yw_v = 0

        my_bat.motion(speed=v_bat, time=time_step, head_p=h_pt_v, body_p=b_pt_v, head_y=h_yw_v, body_y=b_yw_v)
        if fx_hed: my_bat.head_rel.limit_rotations(yaw_limit=0, pitch_limit=0, roll_limit=0)

        previous_bat_x = my_bat.x
        previous_bat_y = my_bat.y
        previous_bat_z = my_bat.z

        previous_bug_x = bug_position[0]
        previous_bug_y = bug_position[1]
        previous_bug_z = bug_position[2]

        # Do some general logging
        logger['time'] = my_bat.clock
        logger['bat_x'] = my_bat.x
        logger['bat_y'] = my_bat.y
        logger['bat_z'] = my_bat.z
        logger['target_x'] = bug_position[0]
        logger['target_y'] = bug_position[1]
        logger['target_z'] = bug_position[2]
        logger['target_v'] = my_bug.get_speed(time_stamps=my_bat.clock)[0]
        logger['target_el'] = target_el
        logger['target_az'] = target_az
        logger['target_d'] = target_dist
        logger['target_el_perceived'] = target_el_perceived
        logger['target_az_perceived'] = target_az_perceived
        logger['out_of_focus'] = out_of_focus
        logger['pulse_interval'] = time_step
        logger['bat_speed'] = v_bat
        logger['h_pt_v'] = h_pt_v
        logger['h_yw_v'] = h_yw_v
        logger['b_pt_v'] = b_pt_v
        logger['b_yw_v'] = b_yw_v

        # Get new pulse interval
        time_step = ipi_profile(target_dist)
        print(target_dist)

    # Prepare return
    export_time_stamps = my_bat.logger['time']

    log_data = logger.export(time_stamps=export_time_stamps)
    bat_data = my_bat.logger.export(time_stamps=export_time_stamps)
    all_data = bat_data.merge(log_data)

    # Add success to data
    all_data['success'] = False
    if interpolated_distance < capture_distance: all_data['success'] = True
    all_data['eta'] = all_data['time'] - numpy.max(all_data['time'])

    # Remove double time entries
    grp = all_data.groupby(['time'])
    all_data = grp.mean()
    all_data.reset_index(inplace=True)

    # Add Settings
    keys = parameters.keys()
    for k in keys: all_data[k] = parameters[k]

    # Add delta time
    # all_data['delta_time'] = numpy.gradient(all_data['time'], edge_order=2)

    # Calculate Gammma
    gammas = calculate_gammas(all_data, 'b')
    all_data['gamma_b'] = gammas
    gammas = calculate_gammas(all_data, 'h')
    all_data['gamma_h'] = gammas
    gammas = calculate_gammas(all_data, 'v')
    all_data['gamma_v'] = gammas

    # Calculate Phi
    time_values = all_data['time']
    phi_real, phi_optimal = calculate_phi(all_data)
    phi_error = numpy.asarray(numpy.rad2deg(phi_real - phi_optimal))
    phi_error = Smoothn.smoothn(phi_error, s=1)[0]
    delta_phi_error = numpy.gradient(phi_error, time_values, edge_order=1)

    all_data['phi_real'] = phi_real
    all_data['phi_optimal'] = phi_optimal
    all_data['phi_error'] = phi_error
    all_data['delta_phi_error'] = delta_phi_error

    # Add profiles to bat object
    my_bat.ipi_profile = ipi_profile
    my_bat.coupling_gn = coupling_gn
    my_bat.coupling_dl = coupling_dl

    # Add head and body acceleration
    all_data['h_pt_a'] = numpy.gradient(all_data['h_pt_v'], time_values, edge_order=1)
    all_data['h_yw_a'] = numpy.gradient(all_data['h_yw_v'], time_values, edge_order=1)
    all_data['b_pt_a'] = numpy.gradient(all_data['b_pt_v'], time_values, edge_order=1)
    all_data['b_yw_a'] = numpy.gradient(all_data['b_yw_v'], time_values, edge_order=1)

    # Return
    return my_bat, all_data


# ###############################################################################################################
# Functions
# ###############################################################################################################

def calculate_gammas(data, dimension='b'):
    target = data.loc[:, ('target_x', 'target_y', 'target_z')]
    bat = data.loc[:, ('bat_x', 'bat_y', 'bat_z')]

    target = target.as_matrix()
    bat = bat.as_matrix()

    vectors = target - bat
    gradient = numpy.gradient(vectors, axis=0)

    gammas = []
    n_steps = gradient.shape[0]
    for n in range(0, n_steps):
        v1 = vectors[n, :]
        g1 = gradient[n, :]
        if dimension == 'h':
            v1[2] = 0
            g1[2] = 0
        if dimension == 'v':
            v1[2] = 0
            g1[2] = 0

        a = angle_between(g1, v1)
        g = numpy.cos(a)
        gammas.append(g)

    # deal with the final missing data point
    # gammas.append(g)
    return gammas


def calculate_phi(data):
    prey = data.loc[:, ('target_x', 'target_y', 'target_z')]
    bat = data.loc[:, ('bat_x', 'bat_y', 'bat_z')]
    time = data['time']
    prey = prey.as_matrix()
    bat = bat.as_matrix()
    time = time.as_matrix()

    prey[:, 2] = 0
    bat[:, 2] = 0

    bat_motion = numpy.gradient(bat, time, axis=0, edge_order=1)
    prey_motion = numpy.gradient(prey, time, axis=0, edge_order=1)
    vectors = prey - bat
    phi_real = []
    phi_optimal = []
    n_steps = bat_motion.shape[0]

    for n in range(0, n_steps):
        prey_velocity_vector = prey_motion[n, :]
        bat_velocity_vector = bat_motion[n, :]
        xx_vector = vectors[n, :]

        prey_velocity = numpy.linalg.norm(prey_velocity_vector)
        bat_velocity = numpy.linalg.norm(bat_velocity_vector)

        prey_velocity_vector = Misc.normalize_vector(prey_velocity_vector)
        bat_velocity_vector = Misc.normalize_vector(bat_velocity_vector)
        xx_vector = Misc.normalize_vector(xx_vector)

        sin_phi = numpy.cross(xx_vector, bat_velocity_vector)
        sin_beta = numpy.cross(xx_vector, prey_velocity_vector)
        beta = angle_between(xx_vector, prey_velocity_vector) * Misc.sign(sin_beta[2])

        real_phi = angle_between(xx_vector, bat_velocity_vector) * Misc.sign(sin_phi[2])
        optimal_phi = numpy.arcsin((prey_velocity * numpy.sin(beta)) / bat_velocity)
        phi_real.append(real_phi)
        phi_optimal.append(optimal_phi)

    phi_real = numpy.asarray(phi_real)
    phi_optimal = numpy.asarray(phi_optimal)

    return phi_real, phi_optimal


def angle_between(v1, v2):
    v1_u = v1 / numpy.linalg.norm(v1)
    v2_u = v2 / numpy.linalg.norm(v2)
    return numpy.arccos(numpy.clip(numpy.dot(v1_u, v2_u), -1.0, 1.0))


def plot_capture3d(data, bat):
    current_clock = bat.clock
    time_stamps = numpy.arange(0, current_clock, 0.1)
    fx = interp1d(data['time'], data['target_x'], bounds_error=False)
    fy = interp1d(data['time'], data['target_y'], bounds_error=False)
    fz = interp1d(data['time'], data['target_z'], bounds_error=False)
    x_inter_insect = fx(time_stamps)
    y_inter_insect = fy(time_stamps)
    z_inter_insect = fz(time_stamps)

    bat.plot_quiver(time_stamps=time_stamps, view='3d')
    pyplot.plot(x_inter_insect, y_inter_insect, z_inter_insect)


def plot_capture(data, bat, view='top'):
    current_clock = bat.clock
    time_stamps = numpy.arange(0, current_clock, 0.1)
    fx = interp1d(data['time'], data['target_x'], bounds_error=False)
    fy = interp1d(data['time'], data['target_y'], bounds_error=False)
    fz = interp1d(data['time'], data['target_z'], bounds_error=False)
    x_inter_insect = fx(time_stamps)
    y_inter_insect = fy(time_stamps)
    z_inter_insect = fz(time_stamps)

    fx = interp1d(data['time'], data['bat_x'], bounds_error=False)
    fy = interp1d(data['time'], data['bat_y'], bounds_error=False)
    fz = interp1d(data['time'], data['bat_z'], bounds_error=False)
    x_inter_bat = fx(time_stamps)
    y_inter_bat = fy(time_stamps)
    z_inter_bat = fz(time_stamps)

    x_lim = Misc.minmax(numpy.concatenate((x_inter_bat, x_inter_insect)))
    y_lim = Misc.minmax(numpy.concatenate((y_inter_bat, y_inter_insect)))
    z_lim = Misc.minmax(numpy.concatenate((z_inter_bat, z_inter_insect)))

    steps = len(time_stamps)
    for i in range(0, steps):
        xs = (x_inter_bat[i], x_inter_insect[i])
        zs = (z_inter_bat[i], z_inter_insect[i])
        ys = (y_inter_bat[i], y_inter_insect[i])
        if view.startswith('t'):
            d, = pyplot.plot(xs, ys, linewidth=0.5, c='grey', alpha=0.5)
        if view.startswith('s'):
            d, = pyplot.plot(xs, zs, linewidth=0.5, c='grey', alpha=0.5)
    if view.startswith('t'): c = pyplot.scatter(x_inter_insect, y_inter_insect, s=5, color='#4286f4')
    if view.startswith('s'): c = pyplot.scatter(x_inter_insect, z_inter_insect, s=5, color='#4286f4')
    results, a, b = bat.plot_quiver(length=0.1, time_stamps=time_stamps, view=view)
    if view.startswith('t'):
        x_lim, y_lim = Misc.scale_ranges(x_lim, y_lim)
        pyplot.title('Top View')
        pyplot.xlim(x_lim)
        pyplot.ylim(y_lim)
    if view.startswith('s'):
        x_lim, z_lim = Misc.scale_ranges(x_lim, z_lim)
        pyplot.title('Side View')
        pyplot.xlim(x_lim)
        pyplot.ylim(z_lim)
    return [a, b, c, d]


def get_gain_setting(cpl_stn):
    # Gain settings
    # distance_range = (0.005, 0.05, 0.1)
    distance_range = (0, 2)
    if cpl_stn == 0:
        k_range = (4, 4)
        t_range = (0.150, 0.150)
        coupling_gn = interp1d(distance_range, k_range, fill_value=(k_range[0], k_range[-1]), bounds_error=False)
        coupling_dl = interp1d(distance_range, t_range, fill_value=(t_range[0], t_range[-1]), bounds_error=False)

    # Gain settings
    if cpl_stn == 1:
        k_range = (6, 6)
        t_range = (0.100, 0.100)
        coupling_gn = interp1d(distance_range, k_range, fill_value=(k_range[0], k_range[-1]), bounds_error=False)
        coupling_dl = interp1d(distance_range, t_range, fill_value=(t_range[0], t_range[-1]), bounds_error=False)

    # Gain settings
    if cpl_stn == 2:
        k_range = (3, 3)
        t_range = (0.150, 0.150)
        coupling_gn = interp1d(distance_range, k_range, fill_value=(k_range[0], k_range[-1]), bounds_error=False)
        coupling_dl = interp1d(distance_range, t_range, fill_value=(t_range[0], t_range[-1]), bounds_error=False)

    # Gain settings
    if cpl_stn == 3:
        k_range = (3, 3)
        t_range = (0.15, 0.15)
        coupling_gn = interp1d(distance_range, k_range, fill_value=(k_range[0], k_range[-1]), bounds_error=False)
        coupling_dl = interp1d(distance_range, t_range, fill_value=(t_range[0], t_range[-1]), bounds_error=False)

    return coupling_gn, coupling_dl


def get_ipi_profile():
    ipi_profile = interp1d([0.87, 2.75], [0.02, 0.1], fill_value=(0.02, 0.1), bounds_error=False)  # Saillant (2007).
    return ipi_profile

#def get_ipi_profile():
#    ipi_profile = interp1d([0.5, 2.75], [0.005, 0.1], fill_value=(0.005, 0.1), bounds_error=False)  # Saillant (2007).
#    return ipi_profile