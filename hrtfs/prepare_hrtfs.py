from scipy.io import loadmat
from matplotlib import pyplot
import numpy
import pickle
import os
import Smoothn
print(os.getcwd())

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
mean_right = numpy.mean(right, 2)
mean_left = numpy.mean(left, 2)
mean_nose = numpy.mean(nose, 2)

left_side_ear = numpy.reshape(mean_left[:,0],(73,1))
right_side_ear = numpy.reshape(mean_left[:,72],(73,1))
left_side_nose = numpy.reshape(mean_nose[:,0],(73,1))
right_side_nose = numpy.reshape(mean_nose[:,72],(73,1))

a = numpy.repeat(left_side_ear,36, axis=1)
b = numpy.repeat(right_side_ear,36, axis=1)
c = numpy.repeat(left_side_nose,36, axis=1)
d = numpy.repeat(left_side_nose,36, axis=1)

mean_left = numpy.concatenate((a, mean_left, b), 1)
mean_right = numpy.concatenate((b, mean_right, a), 1)
mean_nose = numpy.concatenate((c, mean_nose, d), 1)

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
