import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg

# 1) Natural Splines


def natural_coef(x, y, yPrime, h):
    A = scipy.sparse.diags([(1 / 6) * h[1:-1], (1 / 3) *
                            np.add(h[:-1], h[1:]), (1 / 6) * h[1:-1]], [-1, 0, 1])

    y_diff = np.diff(y)
    b_temp = np.divide(y_diff, h)
    b = np.diff(b_temp)

    M = np.zeros(len(x))
    M[1:-1] = scipy.sparse.linalg.cg(A, b)[0]
    return M

# 2) Hermite Splines


def hermite_coef(x, y, yPrime, h):
    A = scipy.sparse.diags([(1 / 6) * h, (1 / 3) *
                            np.add(np.hstack([0, h]), np.hstack([h, 0])), (1 / 6) * h], [-1, 0, 1])

    y_diff = np.diff(y)
    b_temp_temp = np.divide(y_diff, h)
    b_temp = np.diff(b_temp_temp)

    b = np.zeros(len(x))
    b[1: -1] = b_temp
    b[0] = b_temp_temp[0] - yPrime[0]
    b[-1] = yPrime[-1] - b_temp_temp[-1]

    return scipy.sparse.linalg.cg(A, b)[0]

# Find index of given x in the intervall [x0,xn]


def index_finder(num, lst):
    for i in range(1, len(lst)):
        if lst[i - 1] <= num and num <= lst[i]:
            return i
    raise Exception("Element in xx not in range [x0,xn]")


# Determine values of cubic Splines evaluated in points xx

def cubicSpline(x, y, xx, func, yPrime):
    h = np.diff(x)
    M = func(x, y, yPrime, h)
    sol = np.zeros(len(xx))
    for i, num in enumerate(xx):
        j = index_finder(num, x)
        sol[i] = M[j - 1] * ((x[j] - num)**3 / (6 * h[j - 1])) + M[j] * ((num - x[j - 1])**3 / (6 * h[j - 1])) + (y[j - 1] - (1 / 6)
                                                                                                                  * M[j - 1] * (h[j - 1])**2) * ((x[j] - num) / h[j - 1]) + (y[j] - (1 / 6) * M[j] * (h[j - 1])**2) * ((num - x[j - 1]) / h[j - 1])
    return sol


# 3)

def graph_cubicSpline(x, y, func, yPrime, precision=50):
    X = np.linspace(-1, 1, precision)
    Y = cubicSpline(x, y, X, func, yPrime)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    plt.plot(X, Y, 'r')
    plt.scatter(x, y)

    for i in (range(len(x))):
        plt.annotate("({:.2f},{:.2f})".format(
            x[i], y[i]), (x[i], y[i]), textcoords="offset points", xytext=(0, 5), ha='center')

    if func == natural_coef:
        plt.title("Natural Spline Interpolation")
    else:
        plt.title("Hermite Spline Interpolation")


# a)
x1 = [-1, -0.5, -0.25, 0, 0.75, 1]
y1 = [0.6667, 0.2083, -0.3177, -1, -3.6719, -4.6667]
y1Prime = [0, -4]

# b)
x2 = [-1, -0.5, -0.25, 0, 0.75, 1]
y2 = [-1.25, -1, -0.5007, 0, 1.5989, 2.75]
y2Prime = [-3, 7]

graph_cubicSpline(x1, y1, natural_coef, y1Prime)
graph_cubicSpline(x1, y1, hermite_coef, y1Prime)
plt.show()
graph_cubicSpline(x2, y2, natural_coef, y2Prime)
graph_cubicSpline(x2, y2, hermite_coef, y2Prime)
plt.show()


# 4) & 5)
def equiv_cubicSpline(t0, tn, k, y, func, yPrime, precision=50):
    points = np.linspace(t0, tn, k)
    graph = X = np.linspace(t0, tn, precision)
    return cubicSpline(points, y, graph, func, yPrime)

# 6) Interpolation of "L"


points = [(1.4, 5), (3.6, 7.9), (3.2, 9.8), (2.6, 8), (2.55, 5),
          (1.8, 1), (1, 1.8), (2.8, 2.4), (6, 1.1), (9, 2)]

x_val = [i[0] for i in points]
y_val = [i[1] for i in points]
yPrime = [0, 4]


for func in [natural_coef, hermite_coef]:
    x_nat = equiv_cubicSpline(0, 10, 10, x_val, func, yPrime)
    y_nat = equiv_cubicSpline(0, 10, 10, y_val, func, yPrime)

    plt.plot(x_nat, y_nat)
    plt.scatter(x_val, y_val, c='red')

    if func == natural_coef:
        plt.title("Natural Spline Interpolation")
    else:
        plt.title("Hermite Spline Interpolation")

    plt.show()
