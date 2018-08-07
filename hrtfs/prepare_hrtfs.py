from scipy.io import loadmat
from matplotlib import pyplot
import numpy
import pickle
import os

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

buffer_ear = numpy.ones((73, 36, 141)) * ear_min
buffer_nose = numpy.ones((73, 36, 141)) * nose_min

right = numpy.fliplr(left)

left = numpy.concatenate((buffer_ear, left, buffer_nose), 1)
right = numpy.concatenate((buffer_ear, right, buffer_nose), 1)
nose = numpy.concatenate((buffer_nose, nose, buffer_nose), 1)

mean_right = numpy.mean(right, 2)
mean_left = numpy.mean(left, 2)
mean_nose = numpy.mean(nose, 2)

pyplot.figure()
pyplot.subplot(1, 3, 1)
pyplot.imshow(mean_left)
pyplot.title('Left')

pyplot.subplot(1, 3, 2)
pyplot.imshow(mean_right)
pyplot.title('Right')

pyplot.subplot(1, 3, 3)
pyplot.imshow(mean_nose)
pyplot.title('Nose')

hrtf = {'left': left, 'right': right, 'nose': nose, 'freq': freqs}

f = open('pd01.hrtf', 'wb')
pickle.dump(hrtf, f)
f.close()
