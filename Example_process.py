import numpy
from pyBat import Wiegrebe
from matplotlib import pyplot

left_channel = 0
right_channel = 1
left_energy_scale = 1
right_energy_scale = 1.05


cochlea = Wiegrebe.ModelWiegrebe(300000, 40000, 3)

remove_samples = 700

data = numpy.load('000011.npy')

left = data[:, left_channel]
right = data[:, right_channel]

left_removed = left * left_energy_scale
left_removed[:remove_samples] = 0

right_removed = right * right_energy_scale
right_removed[:remove_samples] = 0

pyplot.plot(left)
pyplot.plot(left_removed)
pyplot.show()

pyplot.plot(right)
pyplot.plot(right_removed)
pyplot.show()

result = cochlea.run_model(left)
pyplot.plot(result)
pyplot.show()

m = cochlea.matrix
pyplot.plot(m.T)
pyplot.legend(cochlea.frequencies)
pyplot.show()

