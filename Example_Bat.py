import Bat
import numpy
import Call
import Acoustics

from scipy.interpolate import interp1d

from matplotlib import  pyplot
bat = Bat.Bat()
object_x = numpy.random.randint(-10,10,25)/100 +1
object_y = numpy.random.randint(-10,10,25)/100 + 0.25
object_z = numpy.random.randint(-10,10,25)/100 * 0

objects = numpy.row_stack((object_x, object_y, object_z))
objects = numpy.transpose(objects)

caller = Call.Call('pd01', freq_list= [31000, 35000, 40000])

yaw_magnitude_function = interp1d([0, 1, 5], [180, 90, 60])

for step in range(0, 30):
    bat_cds = bat.world2bat(objects, spherical=True)
    azimuths = bat_cds[:, 0]
    elevations = bat_cds[:, 1]
    distances = bat_cds[:,2]

    result = caller.call(azimuths, elevations, distances)
    left, right, first_delay = Call.process4avoidance(result)
    first_distance = Acoustics.delay2dist(first_delay)

    yaw_sign = 0
    if left < right: yaw_sign = +1
    if right < left: yaw_sign = -1
    print(left,right)

    yaw_magnitude = yaw_magnitude_function(first_distance)

    body_yaw_speed = 3 * float(bat.get_history('a_h', time_stamps=-0.1))
    bat.motion(speed=1, body_y=body_yaw_speed, head_y=yaw_magnitude * yaw_sign, time=0.1)


print('done')
bat.plot_quiver(time_stamps=0.1)
pyplot.scatter(object_x, object_y)
pyplot.xlim([-2,2])
pyplot.ylim([-2,2])
pyplot.show()

#x = numpy.array([1,0,0,0])
#y = numpy.array([0,0,1,0])
#z = numpy.array([0,0,0,0])
#m = Geometry.rotate_points_cart(x,y,z, 90, 0, 0)
#print(m)