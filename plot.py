import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def init_figure():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim3d(0, 1)
    ax.set_ylim3d(0, 1)
    ax.set_zlim3d(0, 1)
    ra = np.linspace(-10, 10, 100)
    c = ra
    d = ra
    c, d = np.meshgrid(c, d)
    X = np.divide(np.exp(2 * c + d), np.multiply(np.multiply(np.exp(c + d), (1 + np.exp(d))), (1 + np.exp(c))))
    Y = np.divide(np.exp(c + 2 * d), np.multiply(np.exp(c + d), np.multiply(1 + np.exp(d), (1 + np.exp(c)))))
    Z = np.divide(np.exp(2 * c + 2 * d), np.multiply(np.multiply(np.exp(c + d), (1 + np.exp(d))), (1 + np.exp(c))))

    ax.plot_surface(X, Y, Z, linewidth=0.2, antialiased=True, alpha=0.5)
    ax.plot(np.linspace(0, 1, 50), np.repeat(0, 50), color='gray')
    ax.plot(np.repeat(0, 50), np.linspace(0, 1, 50), color='gray')
    ax.plot(np.linspace(0, 1, 50), np.linspace(1, 0, 50), color='gray')
    ax.plot(np.repeat(0, 50), np.repeat(0, 50), np.linspace(1, 0, 50), color='gray')
    ax.plot(np.linspace(1, 0, 50), np.repeat(0, 50), np.linspace(0, 1, 50), color='gray')
    ax.plot(np.repeat(0, 50), np.linspace(1, 0, 50), np.linspace(0, 1, 50), color='gray')
    return ax


#ax = init_figure()
scatter_points = dict()


def init_points():
    for color in ['black', 'green', 'yellow', 'red']:
        scatter_points[color] = dict()
        for dim in ['x', 'y', 'z']:
            scatter_points[color][dim] = list()


#init_points()


def plot_distribution(pxy, pxny, pnxy, pnxny, color):
    x = pxny
    y = pnxy
    z = pnxny
    # globalX.append(x)
    # globalY.append(y)
    # globalZ.append(z)
    scatter_points[color]['x'].append(x)
    scatter_points[color]['y'].append(y)
    scatter_points[color]['z'].append(z)
    # ax.scatter([x], [y], [z], color=color, linewidth=1)


def plot_distributions():
    for color in ['black', 'green', 'yellow', 'red']:
        ax.scatter(scatter_points[color]['x'], scatter_points[color]['y'], scatter_points[color]['z'], color=color,
                   linewidth=5, s=1)

    plt.show()