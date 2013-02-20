# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Introduction to Scientific Python

# <markdowncell>

# Astronomers are data scientists. We analyze data and extract all the details we can glean from it. To the last drop. With help from NumPy and SciPy we can do this easily through Python. In a nutshell, here's what NumPy and SciPy are. 
# 
# **NumPy**
# <ul>
# <li> Numerical N-dimensional objects (arrays)</li>
# <li> Powerful broadcasting capability</li>
# <li> Extended and optimized maths capabilities (FFT, etc.)</li>
# <li> Advanced I/O </li>
# </ul>
# 
# **SciPy**
# <ul>
# <li> Advanced maths tools (integration)</li>
# <li> Optimization</li>
# <li> Interpolation</li>
# <li> Statistics</li>
# <li> Multi-dimensional image processing</li>
# <li> Even more advanced I/O</li>
# </ul>

# <headingcell level=3>

# Recap: What is a Python list?

# <codecell>

# A basic list with integers as its elements
list1 = [1, 2, 3, 4]

# A list containing different types of objects
list2 = [list1, range(10), 100, 100.1]

# <markdowncell>

# A list is an object that can store multiple of objects in an ordered sequence, be it lists, dictionaries, integers, floats and so on. This capability makes lists quite powerful as you can use it without worrying about the object types. The only downside is its slow speed of accessing information. Loops especially. 

# <codecell>

abiglist = range(100000) 

def multiply_array(alist, scalar):
    for i, value in enumerate(alist):
        alist[i] = value * scalar

%timeit multiply_array(abiglist, 10)    

# <markdowncell>

# For comparison let's try the same type of operation using NumPy arrays. 

# <codecell>

import numpy as np
abigarray = np.arange(100000, dtype=int)

# Note that we do not need to define a function for this operation
%timeit abigarray * 10

# <markdowncell>

# The NumPy operation is roughly 1000 times faster than the Python loop, which is impressive. The programming is much easier as well. 

# <headingcell level=3>

# What is a NumPy array?

# <markdowncell>

# A NumPy array is an object that can store numbers where each row has the same number of columns. Unlinke lists, it does not have the flexibility to store different types of objects. A standard NumPy array (<code>ndarray</code>) can only accept one type of number. If one element is a float then all the other elements in that array have to be floats. 

# <codecell>

test = np.random.rand(10).astype(str)
print test

# <codecell>

randomness = np.random.rand(100)

new_array = np.column_stack([randomness, randomness.astype(str)])
print type(new_array[0, 0])

# <markdowncell>

# Now let's try to store an integer in this array of zeros that are floats. 

# <codecell>

randomness[10] = int(0)
print type(randomness[10])

# <markdowncell>

# What happened here? The integer was convereted to a float to conform to the array type. So you have to be careful about this when working with arrays. 

# <headingcell level=3>

# Multi-type ndarrays

# <markdowncell>

# It's common that we deal with tables that have different data types per column, like the SQL tables. NumPy can store tables fortunately. Here's an example. 

# <codecell>

a = np.array([(0, 1, 2), (3, 4, 5)], [('x', int), ('y', float), ('z', complex)])

print 'x: ', a['x']
print 'y: ', a['y']
print 'z: ', a['z']

# <markdowncell>

# As you can see, setting up such an array can be cumbersome. There's a package called ATpy ([atpy.github.com](http://atpy.github.com)) that Tom Robitaille and Eli Bressert wrote to simplify this process. We will cover this at the end of the NumPy section.

# <headingcell level=3>

# Array creation and reshaping

# <codecell>

# Creating a list that will be stored as an array
alist = [1, 2, 3]

blist = [[1,2],[3,4]]

aarray = np.array(alist)
barray = np.array(blist)

print aarray.shape
print barray.shape

# <codecell>

print 'alist type is ' + str(type(alist))
print "aarray's type is " + str(type(aarray))

# Let's try this again, but investigating the content of the arrays
blist = [1, 2, 's']
barray = np.array(blist)

print "\nblist's content is " + str(blist)
print "barray's content is " + str(barray)

# <markdowncell>

# Something is askew ... what is it? 

# <markdowncell>

# Making a multi-dimensional array from scratch with all the elements stored as complex numbers:

# <codecell>

array = np.arange(100).reshape(10,10)
print array, '\n'

print array[2:4, 3:6], '\n'

print array[2:4, 3:6].ravel()

# <markdowncell>

# The dtype option can take in `float`, `float32`, `float64`, `int`, and more. Now, let's take the array and shuffle the elements around a bit. 

# <codecell>

# Flattening the array
print 'Number of dimensions: ', array.ravel().shape
print array.ravel()

# <codecell>

# Reshaping an array
flat_array = array.ravel()
new_array = flat_array.reshape(10,10)
print 'Number of dimensions: ', new_array.shape
print new_array

# <codecell>

# If you want to transpose an array you can simply do
print new_array.transpose()

# <headingcell level=3>

# Operations

# <markdowncell>

# When executing operations on an array, it is applied elementwise by default. This is called **broadcasting** in NumPy terms. If needed, linear algrebra operatoins can be called instead. 

# <codecell>

# Some basic operations
array = np.arange(10)
print 'Example 1: ', array + 1

# Is the same as 
a = np.repeat(1,10)
print 'Example 2: ', array + a

# <codecell>

# The same goes for the other operators like ...
#print array * 2.0  # Multiplication
print array ** 2   # Exponentiation
#print array / 2.0    # Division
# ... and so on
#print array.max()
#print array.std()
#print array.var()
#print array.mean()

# <headingcell level=2>

# Slicing and dicing

# <markdowncell>

# Accessing elements in a list is simple enough, but if you have a multi-dimenstional list it gets tricky.

# <codecell>

alist = [[1, 2],[3, 4]]

# If we want to access the element three we would do
print alist[1][0], '\n'

# With NumPy it's easier and more intuitive
arr = np.array(alist)
print arr[1, 0], '\n'

#This also gives us the ability to subsample an array with ease.
arr = np.arange(100).reshape(10,10)
print arr, '\n'

print arr[1:3, 2:4]

# <markdowncell>

# We can also inject new values into the array using the the indexing method. 

# <codecell>

arr[1:3, 2:7] = -99
print arr

# <headingcell level=2>

# Indexing with boolean arguments

# <markdowncell>

# When I review other people's code the biggest component that slows them down is how the arrays are modified. Imagine that you have an *N* dimension array where you have random values that range between 0 and 1 for each element. If an element is less than 0.5 then you want that element to be set to 0. For loop fans, this would be the obvious solution. 

# <codecell>

import numpy as np
arr = np.random.rand(100000)

def filter(arr):
    for i, val in enumerate(arr):
        if val < 0.5:
            arr[i] = 0
    return arr

# <markdowncell>

# The time it takes for this operation is 840 ms. With NumPy we are supposed to discard of loops when we can, so if you do the following then you will save a lot of execution time. 

# <codecell>

import numpy as np
arr = np.random.rand(10)
arr[arr < 0.5] = 0
print arr

# <markdowncell>

# This method takes only 10.4 ms, which is a factor of 80 times faster than the the previous example. The example above is very simple in terms of boolean arguments, so what can we do when it becomes more complicated? 

# <codecell>

True and True

# <codecell>

import numpy as np
arr = np.random.rand(5)
index1 = arr < 0.8
index2 = arr > 0.2
index3 = arr == 0.9

print index1
print index2
print index3
print ~((index1 & index2) | index3)

arr[~((index1 & index2) | index3)] = -1

# <markdowncell>

# We filtered the array for values in between 0.2 and 0.8, with the exception of any element that has a value of 0.9. Any other number is set to -1. With NumPy arrays we can use the `&` symbol as the **and** operator and the `|` symbol as the **or** operator. The tilde (`~`) symbol acts as the **not** operator. You can alternatively just return a new array without the elements that do not meet our criteria.

# <codecell>

arr2 = arr[(index1 & index2) | index3]

# <markdowncell>

# This type of slicing and indexing to modify arrays can work on whatever array structure you can come up with, whether it's 1-D, 3-D, or higher.
# 
# As a quick side note: Using the `np.where` function is another way of doing the examples above, but in some cases it will be a bit slower.

# <headingcell level=3>

# Input and output

# <codecell>

import numpy as np

# Loading data
data = np.loadtxt('testfile.txt')

# Print out information about the array
print data.shape, '\n'
print data

# <codecell>

# Saving as a NumPy pickle, which binary
np.save('test.npy', data) 
np.savez('test.npyz', data) # Compressed binary
np.savetxt('test.txt', data)

# <headingcell level=3>

# ATpy: Simplifying data access

# <codecell>

import urllib2
import atpy

# Downloading some 2MASS data on the Orion Nebula Cluster
url = 'http://irsa.ipac.caltech.edu/workspace/TMP_Rb7N1N_4880/Gator/irsa/7011/fp_2mass.fp_psc7011.tbl'
open('fp_2mass.fp_psc7011.tbl', 'w').write(urllib2.urlopen(url).read())

# Reading in the table
t = atpy.Table('fp_2mass.fp_psc7011.tbl')

# See what columns are in the table and what data types they are
t.describe()
t.shape

# <codecell>

# Access some columns for a plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(t['ra'], t['dec'])
plt.show(fig)

# <codecell>

# Checking magnitude range in K_s band data
print np.sort(t['k_m'])

# Filter the table to show only bright stars
new_t = t.where(t['k_m'] < 12)

# Checking out number of remaining sources
print new_t.shape

t.write('filtered_table.tbl', overwrite=True)

# Because the table has no name
t.table_name = '2MASS_data'
t.write('filtered_table.fits', overwrite=True)

# <headingcell level=3>

# Excercise

# <markdowncell>

# Create a data array using `np.random.normal`, with dimensions of 20 rows and 4 columns and do the following:
# 
# <ul>
#     <li>Compute the standard deviation</li>
#     <li>Compute the maximum and minimum values</li>
#     <li>Filter out all values above 1 and below -1</li>
#     <li>Save the new array using np.savetxt</li>
# </ul>

# <codecell>

# Solution

array = np.random.normal(size=20*4).reshape(20,4)
print np.std(array)
print array.max()
print array.min()

array[~((array < 1) & (array > -1))] = np.nan
print '\n', array
np.savetxt('exercise_solution.txt', array)

# <headingcell level=2>

# Plotting with Matplotlib

# <markdowncell>

# We're going to cover a few basic plotting routines in Matplotlib. 
# 
# <ul>
#     <li>Line plot</li>
#     <li>Scatter</li>
#     <li>Histogram</li>
# </ul>
# 
# Matplotlib can be run either interactively or static. The dynamic option is good for exploration and static is good making publication ready plots. Then you save them and run them when you need to. 

# <headingcell level=4>

# Line plot

# <codecell>

import numpy as np
import matplotlib.pyplot as plt

# Creating data
x = linspace(0, 5 * np.pi, 1000)
y = np.cos(x)

# Initiating plot figure 
fig = figure()
ax = fig.add_subplot(111)

# Plotting cosine function
ax.plot(x, y)

plt.show(fig)

# <codecell>

import numpy as np
import matplotlib.pyplot as plt

# Creating data
x = linspace(0, 5 * np.pi, 1000)
y1 = np.cos(x)
y2 = np.cos(x ** 2)

# Initiating figure
fig = figure(figsize=(10, 6))

# Adding subplots
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

# Plotting data
ax1.plot(x, y1)
ax2.plot(x, y2, color='red')

plt.show(fig)

# <codecell>

import numpy as np
import matplotlib.pyplot as plt

# Creating data
x = linspace(0, 5 * np.pi, 1000)
fig = figure(figsize=(10, 6))

# Looping data to create an alpha effect
for i in xrange(1,100):
    y = np.cos(x + i*0.1)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y, color='black', alpha=1.0 / i, linestyle='--')

#ax.set_xlim(left=0 - 1 , right=x.max() + 1)
#ax.set_ylim(top=-1 - 1 , bottom=y.max() + 1)
plt.show(fig)

# <headingcell level=4>

# Scatter plot

# <codecell>

import numpy as np
import matplotlib.pyplot as plt

# Setting up figure properties
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)

# Generating fake star cluster data
cluster_x = np.random.normal(size=100) 
cluster_y = np.random.normal(size=100)

# Generating fake field star data
field_x = np.random.uniform(low=-3, high=3, size=200)
field_y = np.random.uniform(low=-3, high=3, size=200)

# Scattter plot
ax.scatter(field_x, field_y, edgecolor='black', facecolor='none', s=50)
ax.scatter(cluster_x, cluster_y, marker='^', c='red', s=100, alpha=0.5)

# Setting aspect ratio between X and Y axis to be equal
ax.set_aspect('equal')
ax.set_xlim(left=-3, right=3)
ax.set_ylim(bottom=-3, top=3)

# Adding labels
ax.set_xlabel('x offset')
ax.set_ylabel('y offset')

# Showing the figure
plt.show(fig)

# <headingcell level=4>

# Histogram

# <codecell>

import numpy as np
import matplotlib.pyplot as plt

dist = np.random.normal(size=100000)

# Setting up figure properties
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

# Histogram
ax.hist(dist, bins=100, histtype='stepfilled', alpha = .5, normed=True)

show(fig)

# <headingcell level=2>

# SciPy: Optmiziation, interpolation and statistics

# <headingcell level=3>

# Optimization

# <markdowncell>

# SciPy provides a nice function called `curve_fit` to solve minimization problems easily. Below are two simple examples, where the first is a solution to a a linear function and the second to a Gaussian function.

# <codecell>

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# A linear function that we will fit to data
def line(x, a, b):
    return a * x + b

# Fake clean data
x = np.linspace(0, 10, 100)
y = line(x, 1, 2)

# Adding noise to the fake data
yn = y + 0.9 * np.random.normal(size=len(x))

# Executing curve_fit on noisy data
popt, pcov = curve_fit(line, x, yn)

# popt returns the best fit values for parameters of the given model (func)
print popt, '\n'

# pcov returns the covariance matrix of the fit. 
print pcov, '\n'

# If you want the standard deviation of the fit then
print np.sqrt(pcov.diagonal())

# <codecell>

# Plotting the output for visual inspection
ym = line(x, popt[0], popt[1])
fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111)
ax.plot(x, y, c='k', label='Function')
ax.scatter(x, yn)
ax.plot(x, ym, c='r', label='Best fit')
ax.legend(loc='upper left')
plt.show(fig)

# <codecell>

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# A Gaussian function that we will fit to data
def gauss(x, a, b, c):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))

# Fake clean data
x = np.linspace(0, 10, 100)
y = gauss(x, 1, 3, 2)

# Adding noise to the fake data
yn = y + 0.2 * np.random.normal(size=len(x))

# Executing curve_fit on noisy data
popt, pcov = curve_fit(gauss, x, yn)

# popt returns the best fit values for parameters of the given model (func)
print popt, '\n'

# pcov returns the covariance matrix of the fit. 
# Rule of thumb: if diagonal is negative = bad, if positive = good.
print pcov, '\n'

# If you want the standard deviation of the fit then
print np.sqrt(pcov.diagonal())

# <codecell>

# Plotting the output for visual inspection
ym = gauss(x, popt[0], popt[1], popt[2])
fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111)
ax.plot(x, y, c='k', label='Function')
ax.scatter(x, yn)
ax.plot(x, ym, c='r', label='Best fit')
ax.legend(loc='upper left')

# <headingcell level=3>

# 1D interpolation

# <codecell>

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Creating fake data.
x = np.linspace(0, 10 * np.pi, 20)
y = np.cos(x)

# 1D interpolation
fl = interp1d(x, y, kind='linear')
fq = interp1d(x, y, kind='quadratic')

# x.min and x.max are the boundaries
xint = np.linspace(x.min(), x.max(), 1000)
yintl = fl(xint)
yintq = fq(xint)

# <codecell>

# Plotting output
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111)
ax.plot(xint, yintl, label='Linear')
ax.plot(xint, yintq, label='Quadratic')
ax.scatter(x, y, marker='o', edgecolor='red', facecolor='none', s=100)
ax.legend(loc='lower right')
ax.set_xlim(0, 10 * np.pi)
ax.set_ylim(-2, 2)
plt.show(fig)

# <headingcell level=3>

# Statistics: Distribution and comparison

# <markdowncell>

# Here we'll setup two samples of data and then compare them to see if they are similar using the KS test. First we setup the normal distribution.<br> 
# 
# $$\textrm{PDF} = \exp\left(\frac{-x^2/2}{\sqrt{2\;\pi}}\right)$$

# <headingcell level=4>

# Normal distribution

# <codecell>

import numpy as np
from scipy.stats import norm

# Setup the sample range
x = np.linspace(0, 10, 1000)

# Here set up the parameters for the normal distribution.
# where loc is the mean and scale is the standard deviation.
dist = norm(loc=5, scale=1)

# Here we draw out 1000 random values from
norm_sample = dist.rvs(1000)

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111)
ax.hist(norm_sample, bins=20, histtype='stepfilled')
plt.show(fig)

# <headingcell level=4>

# Lognormal distribution

# <codecell>

import numpy as np
from scipy.stats import lognorm

# Setup the sample range
x = np.linspace(0, 10, 1000)

# Here set up the parameters for the lognormal distribution.
stddev = 0.9
mean = 0
dist = lognorm([stddev],loc=mean)

# Here we draw out 500 random values from
lognorm_sample = dist.rvs(1000)

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111)
ax.hist(lognorm_sample, bins=20, histtype='stepfilled')
plt.show(fig)

# <codecell>

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111)
ax.hist(np.log10(lognorm_sample), bins=20, histtype='stepfilled')
plt.show(fig)

# <headingcell level=4>

# Distribution comparison

# <codecell>

import numpy as np
from scipy import stats

D, pvalue = stats.ks_2samp(norm_sample, lognorm_sample)
print('Comparing normal to normal distribution')
print('D = ' + str(D))
print('P-value = ' + str(pvalue))

# If KS test is good then you should get a high value (Max =1.0).
# If KS test is bad then you should get a low value (Min = 0.0) 

# <codecell>

D, pvalue = stats.ks_2samp(norm_sample, norm_sample)
print('Comparing normal to normal distribution')
print('D = ' + str(D))
print('P-value = ' + str(pvalue))

# <codecell>

D, pvalue = stats.ks_2samp(norm_sample, np.log10(lognorm_sample) + 5)
print('Comparing normal to log10(lognormal) + 5 distribution')
print('D = ' + str(D))
print('P-value = ' + str(pvalue))

