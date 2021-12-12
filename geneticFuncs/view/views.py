from deap import benchmarks
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from django.shortcuts import render
import numpy as np
import math
from io import BytesIO
import base64

##############################################################################
##############################################################################

# ackley_function


def ackley_function_range(x_range_array):
    # returns an array of values for the given x range of values
    value = np.empty([len(x_range_array[0])])
    for i in range(len(x_range_array[0])):

        # returns the point value of the given coordinate
        part_1 = -0.2 * \
            math.sqrt(0.5*(x_range_array[0][i]*x_range_array[0]
                      [i] + x_range_array[1][i]*x_range_array[1][i]))
        part_2 = 0.5 * \
            (math.cos(2*math.pi*x_range_array[0][i]) +
             math.cos(2*math.pi*x_range_array[1][i]))

        value_point = math.exp(1) + 20 - 20*math.exp(part_1) - math.exp(part_2)
        value[i] = value_point
    # returning the value array
    return value


# def plot_ackley_general():
# this function will plot a general ackley function just to view it.
limit = 1000  # number of points
# common lower and upper limits for both x1 and x2 are used
lower_limit = -5
upper_limit = 5
# generating x1 and x2 values
x1_range = [np.random.uniform(lower_limit, upper_limit)
            for x in range(limit)]
x2_range = [np.random.uniform(lower_limit, upper_limit)
            for x in range(limit)]
# This would be the input for the Function
x_range_array = [x1_range, x2_range]
# generate the z range
z_range = ackley_function_range(x_range_array)
# plotting the function
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x1_range, x2_range, z_range, label='Ackley Function')
# plt.show()
ackley_imgA = BytesIO()
plt.savefig(ackley_imgA, format='png')
plt.close()
ackley_imgA.seek(0)
ackley_urlA = base64.b64encode(ackley_imgA.getvalue()).decode('utf8')


def plot_ackley(x1_range, x2_range):
    x_range_array = [x1_range, x2_range]
    z_range = ackley_function_range(x_range_array)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x1_range, x2_range, z_range, label='Ackley Function')


##############################################################################
##############################################################################

# restringing_graph

X = np.linspace(-5.12, 5.12, 100)
Y = np.linspace(-5.12, 5.12, 100)
X, Y = np.meshgrid(X, Y)

Z = (X**2 - 10 * np.cos(2 * np.pi * X)) + \
    (Y**2 - 10 * np.cos(2 * np.pi * Y)) + 20

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap=cm.nipy_spectral, linewidth=0.08,
                antialiased=True)
# plt.show()
restringing_img = BytesIO()
plt.savefig(restringing_img, format='png')
plt.close()
restringing_img.seek(0)
restringing_url = base64.b64encode(restringing_img.getvalue()).decode('utf8')


##########################################################################
##########################################################################

# griewank

def h1_arg0(sol):
    return benchmarks.h1(sol)[0]


fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-25, 25, 0.5)
Y = np.arange(-25, 25, 0.5)
X, Y = np.meshgrid(X, Y)
Z = np.fromiter(map(h1_arg0, zip(X.flat, Y.flat)), dtype=np.float,
                count=X.shape[0]*X.shape[1]).reshape(X.shape)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                norm=LogNorm(), cmap=cm.jet, linewidth=0.2)

plt.xlabel("x")
plt.ylabel("y")

# plt.show()
griewank_img = BytesIO()
plt.savefig(griewank_img, format='png')
plt.close()
griewank_img.seek(0)
griewank_url = base64.b64encode(griewank_img.getvalue()).decode('utf8')


##########################################################################
##########################################################################


# Quadric


# 100 linearly spaced numbers
x = np.linspace(0, 1, 10)
eta1 = 0.05
eta2 = 0.1
lmd1 = 2.8
# the function, which is y = x^2 here
y = 0.5*(-(1/3)*eta1*x**2 + np.sqrt(((1/3)*eta1*x**2)**2-4*(3*x**2*2.8**2-1)))

# setting the axes at the centre
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.plot(x, y, 'r')

# plt.show()
Quadric_img = BytesIO()
plt.savefig(Quadric_img, format='png')
plt.close()
Quadric_img.seek(0)
Quadric_url = base64.b64encode(Quadric_img.getvalue()).decode('utf8')


############################################################################################

# view on http://127.0.0.1:8000/


def home(request):
    return render(request, 'home.html', {"restringing_url": restringing_url, "ackley_urlA": ackley_urlA, "griewank_url": griewank_url, "Quadric_url": Quadric_url})
