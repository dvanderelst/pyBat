import numpy
from matplotlib import pyplot
from sklearn import linear_model
import shutil
import os
from mpl_toolkits.basemap import Basemap


def plot_map(az, el, z, levels):
    mmap = Basemap(projection='hammer', lat_0=0, lon_0=0)
    x, y = mmap(az, el)
    mmap.contourf(x, y, z, tri=True, cmap='inferno', levels=levels)
    parallels = numpy.arange(-90, 90, 60)
    mmap.drawparallels(parallels)
    meridians = numpy.arange(-180, 180, 60)
    mmap.drawmeridians(meridians)


def clear_folder(folder):
    shutil.rmtree(folder)
    os.makedirs(folder)


def plot_robust(result):
    x_in = result['x_in']
    y_in = result['y_in']
    x_out = result['x_out']
    y_out = result['y_out']
    x_all = result['x_all']
    y_all = result['y_all']
    pyplot.subplot(2, 2, 1)
    pyplot.scatter(x_in, y_in, color='yellowgreen', marker='.', label='Inliers')
    pyplot.scatter(x_out, y_out, color='gold', marker='.', label='Outliers')
    pyplot.legend(loc='lower right')
    pyplot.xlabel("Input")
    pyplot.ylabel("Response")
    pyplot.subplot(2, 2, 2)
    pyplot.scatter(x_in, y_in, color='yellowgreen', marker='.', label='Inliers')
    pyplot.subplot(2, 2, 3)
    pyplot.hist2d(x_all, y_all, bins=20)
    pyplot.colorbar()


def robust(x, y, do_plot=False):
    x = numpy.array(x).reshape(-1, 1)
    y = numpy.array(y).reshape(-1, 1)
    min_inliers = int(len(x) * 0.90)

    model = linear_model.RANSACRegressor(max_trials=10000, stop_n_inliers=min_inliers)
    model.fit(x, y)
    inlier_mask = model.inlier_mask_
    outlier_mask = numpy.logical_not(inlier_mask)

    x_all = x.flatten()
    y_all = y.flatten()
    x_in = x[inlier_mask].flatten()
    y_in = y[inlier_mask].flatten()
    x_out = x[outlier_mask].flatten()
    y_out = y[outlier_mask].flatten()
    proportion = numpy.nanmean(inlier_mask)

    correlation = numpy.corrcoef(x_in, y_in)
    correlation = correlation[0, 1]
    results = {}
    results['corr'] = correlation
    results['x_in'] = x_in
    results['y_in'] = y_in
    results['x_all'] = x_all
    results['y_all'] = y_all
    results['x_out'] = x_out
    results['y_out'] = y_out
    results['model'] = model
    results['prop'] = proportion

    if do_plot: plot_robust(results)
    return results


def robust_n(x, y, n, do_plot=False):
    best_result = None
    best_correlation = 0
    for replication in range(0, n):
        current_result = robust(x, y)
        current_correlation = current_result['corr']
        if abs(current_correlation) > abs(best_correlation):
            best_correlation = current_correlation
            best_result = current_result
    if do_plot: plot_robust(best_result)
    return best_result


def gforce(velocity, turning_rate):
    rate = numpy.deg2rad(turning_rate)
    g = (velocity * rate) / 9.81
    return g


def turning_rate(velocity, gforce=2):
    rate = (gforce * 9.81) / velocity
    rate = numpy.rad2deg(rate)
    return rate


def mat2array(matrix):
    array = numpy.array(matrix, dtype='f')
    array = numpy.squeeze(array)
    if array.shape == (): array = numpy.reshape(array, (1,))
    return array


def normalize_vector(vector):
    norm = numpy.linalg.norm(vector)
    if norm == 0:return  vector
    x = vector / norm
    return x


def angle_between(v1, v2):
    v1 = numpy.reshape(v1, (1, -1))
    v2 = numpy.reshape(v2, (-1, 1))
    v1_u = normalize_vector(v1)
    v2_u = normalize_vector(v2)
    angle = numpy.arccos(numpy.clip(numpy.dot(v1_u, v2_u), -1.0, 1.0))
    angle = float(angle)
    angle = numpy.rad2deg(angle)
    if numpy.isnan(angle): return 90
    return angle


def signed_vector_angle(p1, p2):
    ang1 = numpy.arctan2(*p1[::-1])
    ang2 = numpy.arctan2(*p2[::-1])
    return numpy.rad2deg((ang1 - ang2) % (2 * numpy.pi))


def inter_distance(start1, end1, start2, end2):
    start1 = numpy.array(start1, dtype='f')
    start2 = numpy.array(start2, dtype='f')

    end1 = numpy.array(end1, dtype='f')
    end2 = numpy.array(end2, dtype='f')

    norm1 = numpy.linalg.norm(end1 - start1)
    norm2 = numpy.linalg.norm(end2 - start2)

    n = (numpy.ceil(max(norm1, norm2)) + 1) * 2

    x1 = numpy.linspace(start1[0], end1[0], n)
    y1 = numpy.linspace(start1[1], end1[1], n)
    z1 = numpy.linspace(start1[2], end1[2], n)

    x2 = numpy.linspace(start2[0], end2[0], n)
    y2 = numpy.linspace(start2[1], end2[1], n)
    z2 = numpy.linspace(start2[2], end2[2], n)

    distance = (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2
    distance = numpy.sqrt(distance)
    distance = numpy.min(distance)
    return distance


def append2csv(data, path, sep=","):
    import os
    if not os.path.isfile(path):
        data.to_csv(path, mode='a', index=False, sep=sep)
    else:
        data.to_csv(path, mode='a', index=False, sep=sep, header=False)


def scale_ranges(a, b, zoom_out=1.25):
    rng_a = numpy.ptp(a)
    rng_b = numpy.ptp(b)
    mean_a = numpy.mean(a)
    mean_b = numpy.mean(b)
    if rng_b > rng_a: a = (b - mean_b) + mean_a
    if rng_a > rng_b: b = (a - mean_a) + mean_b
    mean_a = numpy.mean(a)
    mean_b = numpy.mean(b)
    a = ((a - mean_a) * zoom_out) + mean_a
    b = ((b - mean_b) * zoom_out) + mean_b
    return a, b


def minmax(array):
    mn = numpy.nanmin(array)
    mx = numpy.nanmax(array)
    rng = numpy.array((mn, mx), dtype='f')
    return rng


def unwrap(angles):
    radians = numpy.deg2rad(angles)
    radians = numpy.unwrap(radians)
    angels = numpy.rad2deg(radians)
    return angels


def iterable(x):
    if isinstance(x, str): return False
    try:
        for t in x:
            break
        return True
    except:
        return False


def isstr(x):
    return isinstance(x, str)


def nan_array(shape):
    return numpy.full(shape, numpy.nan)


def rand_range(min_value, max_value, shape):
    y = numpy.random.random(shape)
    y = y * (max_value - min_value)
    y = y + min_value
    return y


def unit_vector(norm=1):
    return numpy.array([[0], [0], [1]], dtype='f') * norm


def closest(array, value):
    idx = (numpy.abs(array - value)).argmin()
    return array[idx], idx


def angle_arrays(az_range=180, el_range=90, step=2.5, grid=True):
    az_range = abs(az_range)
    el_range = abs(el_range)
    az = numpy.arange(-az_range, az_range + 0.001, step)
    az = numpy.transpose(az)
    el = numpy.arange(-el_range, el_range + 0.001, step)
    if not grid: return az, el
    az, el = numpy.meshgrid(az, el)
    return az, el


def random_cds(n, min_value, max_value, y_zero=True):
    points = numpy.random.rand(n, 3)
    r = max_value - min_value
    points = (points * r) + min_value
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    if y_zero: y = numpy.zeros(x.shape)
    return x, y, z


def corr2cov(sigma, corr):
    sigma = numpy.array(sigma, dtype='f')
    corr = numpy.array(corr, dtype='f')
    cov = corr * (numpy.transpose(sigma) * sigma)
    return cov


def almost_equal(a, b, threshold):
    diff = abs(a - b)
    if diff <= threshold: return True
    return False


def sign(x):
    if x < 0: return -1
    if x > 0: return 1
    if x == 0: return 0


def values2labels(lst):
    labels = []
    for x in lst:
        label = 'Condition ' + str(x+1)
        labels.append(label)
    return labels
    #    if x == 0:
    #        labels.append('Fixed')
    #    else:
    #        labels.append(str(x) + ' deg/s$^2$')
    #return labels