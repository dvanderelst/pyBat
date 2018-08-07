import matplotlib
from matplotlib import pyplot


def set_style():
    pyplot.style.use('default')
    pyplot.style.use('seaborn-darkgrid')
    matplotlib.rcParams['font.weight'] = 200
    matplotlib.rcParams['font.family'] = "sans-serif"
    matplotlib.rcParams['font.sans-serif'] = "DejaVuSans"
    matplotlib.rcParams['font.weight'] = 200

    pyplot.rcParams['savefig.facecolor'] = 'white'
# Then, "ALWAYS use sans-serif fonts"
