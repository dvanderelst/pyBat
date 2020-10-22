import glob
import natsort
import os
import numpy

path = "/home/dieter/Dropbox/PythonRepos/DAQgui/data"

def readfolder(folder):
    full_path = os.path.join(path, folder, '*.npy')
    files = glob.glob(full_path)
    files = natsort.natsort.natsorted(files)
    n = len(files)
    data = numpy.load(files[0])
    shape = list(data.shape)
    shape.append(n)
    all_data = numpy.zeros(shape)
    for i in range(n):
        data = numpy.load(files[i])
        all_data[:,:,:,:,i] = data
    return all_data


