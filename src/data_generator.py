import numpy
from matplotlib.pyplot import rcParams


def sin2PiX(x):
    return numpy.sin(2 * numpy.pi * x)


def generateData(func=sin2PiX, N=5000, sigma=1.0, start=-2.0, stop=2.0):
    """Generate data with random Gaussian noise whose expectation is 0.

    Parameters:
        func -- the function used to generate data points(default sin(2pi x))
        N -- the number of data points you want to generate (default 50)
        sigma -- the standard deviation of the Gaussian noise (default 1.0)
        start -- Start of interval. (default -2.0)
        stop -- End of interval. (default 2.0)
    Returns:
        A dict {'xArray': array, 'tArray': array} in which each array is
        the data points' X-axis and Y-axis
    """
    # 用来正常显示中文标签
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['SimHei', 'Helvetica', 'Calibri']
    # 用来正常显示负号
    rcParams['axes.unicode_minus'] = False

    # For N points, there is only have N -1 intervals
    xArray = numpy.arange(start, stop + 0.001, (stop - start)/(N - 1))
    yArray = numpy.array(list(map(func, xArray)))

    # Random Gaussian noise
    noise = numpy.random.normal(0.0, sigma, N)
    # Show the noise
    # import matplotlib.pyplot as plt
    # count, bins, ignored = plt.hist(noise, 30, density=True)
    # plt.plot(bins, 1/(sigma * numpy.sqrt(2 * numpy.pi)) * 
    #     numpy.exp( - (bins - 0)**2 / (2 * sigma**2) ),
    #         linewidth=2, color='r')
    # plt.show()

    yArray += noise
    return {'xArray': xArray, 'yArray': yArray}
