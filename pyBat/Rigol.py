import numpy
from os import path
from scipy.io.wavfile import write
from pyBat import Signal

def readCSV(filename):
    f = open(filename)
    data = f.readlines()
    parameters = data[1]
    parameters = parameters.rstrip(',')
    parameters = parameters.split(',')
    fs = round(1/ float(parameters[-1]))
    data = data[2:]
    signal = []
    for x in data:
        x = x.rstrip('\n')
        x = x.rstrip(',')

        point = x.split(',')
        signal.append(point[1:])
    signal = numpy.array(signal)
    signal = signal.astype('float')
    return signal, fs


def csv2wav(filename, output = None):
    parts = path.splitext(filename)
    base = parts[0]
    if output is None: output = base + '.wav'
    data, fs = readCSV(filename)
    write(output, rate=fs, data=data)


from pyBat import Signal
from matplotlib import pyplot

#csv2wav('/home/dieter/Desktop/NewFile1.csv')
f = '/media/dieter/USB20FD/NewFile8.csv'
data, fs = readCSV(f)
signal = data[:,0]
out = data[:,0]
bp = Signal.BandBassFilter(32000, 42000, fs, order=5)
filtered_signal = bp.run(signal)
Signal.simple_spectrogam(filtered_signal, sample_rate=fs, n=2**6)
pyplot.plot(signal)
pyplot.plot(filtered_signal)
#pyplot.plot(out)
pyplot.show()