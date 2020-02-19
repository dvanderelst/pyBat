from scipy.io import loadmat
from matplotlib import pyplot
import numpy
import pickle
import os
print(os.getcwd())

#
# HRTFS are assumed to be -180, 180 and -90 90. In steps of 2.5 degrees.
#

###############################
# Standard Phillostomus HRTF
###############################s

pd = loadmat('PhilloTot.mat')
data = pd['PhilloTot']
left = data['ears'][0, 0]
nose = data['nose'][0, 0]
freqs = data['freqs'][0, 0]
freqs = numpy.squeeze(freqs)

ear_min = numpy.min(left)
nose_min = numpy.min(nose)

right = numpy.fliplr(left)
nose = (nose + numpy.fliplr(nose))/2


left_side_ear = numpy.reshape(left[:,0,:],(73,1,141))
right_side_ear = numpy.reshape(left[:,72,:],(73,1,141))
left_side_nose = numpy.reshape(nose[:,0,:],(73,1,141))
right_side_nose = numpy.reshape(nose[:,72,:],(73,1,141))

a = numpy.repeat(left_side_ear,36, axis=1)
b = numpy.repeat(right_side_ear,36, axis=1)
c = numpy.repeat(left_side_nose,36, axis=1)
d = numpy.repeat(left_side_nose,36, axis=1)

left = numpy.concatenate((a, left, b), 1)
right = numpy.concatenate((b, right, a), 1)
nose = numpy.concatenate((c, nose, d), 1)

mean_left = numpy.mean(left, axis=2)
mean_right = numpy.mean(right, axis=2)
mean_nose = numpy.mean(nose, axis=2)

pyplot.figure()
pyplot.subplot(2, 2, 1)
pyplot.imshow(mean_left)
pyplot.title('Left')

pyplot.subplot(2, 2, 2)
pyplot.imshow(mean_right)
pyplot.title('Right')

pyplot.subplot(2, 2, 3)
pyplot.imshow(mean_nose)
pyplot.title('Nose')

hrtf = {'left': left, 'right': right, 'nose': nose, 'freq': freqs}

f = open('pd01.hrtf', 'wb')
pickle.dump(hrtf, f)
f.close()
pyplot.show()
