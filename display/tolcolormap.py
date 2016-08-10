#!/usr/bin/env python2.7
# encoding: utf-8
#
# matplotlib-ref-density.py -- matplotlib example script
# Copyright (C) 2011 Tim van Werkhoven (t.i.m.vanwerkhoven@xs4all.nl)
#
# This work is licensed under the Creative Commons Attribution-Share Alike 
# 3.0 Unported License. To view a copy of this license, visit 
# http://creativecommons.org/licenses/by-sa/3.0/ or send a letter to Creative 
# Commons, 171 Second Street, Suite 300, San Francisco, California,94105, USA.

# import pylab as plt
import numpy as N
import matplotlib

# Make colormap based on Paul Tol's best visibility gradients. See 
# <http://www.sron.nl/~pault/> for more info on these colors. Also see
# <http://matplotlib.sourceforge.net/api/colors_api.html>
# and <http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps> on some 
# matplotlib examples

# Deviation around zero colormap (blue--red)
cols = []
for x in N.linspace(0,1, 256):
	rcol = 0.237 - 2.13*x + 26.92*x**2 - 65.5*x**3 + 63.5*x**4 - 22.36*x**5
	gcol = ((0.572 + 1.524*x - 1.811*x**2)/(1 - 0.291*x + 0.1574*x**2))**2
	bcol = 1/(1.579 - 4.03*x + 12.92*x**2 - 31.4*x**3 + 48.6*x**4 - 23.36*x**5)
	cols.append((rcol, gcol, bcol))

cm_plusmin = matplotlib.colors.LinearSegmentedColormap.from_list("PaulT_plusmin", cols)

# Linear colormap (white--red)
from scipy.special import erf

cols = []
for x in N.linspace(0,1, 256):
	rcol = (1 - 0.392*(1 + erf((x - 0.869)/ 0.255)))
	gcol = (1.021 - 0.456*(1 + erf((x - 0.527)/ 0.376)))
	bcol = (1 - 0.493*(1 + erf((x - 0.272)/ 0.309)))
	cols.append((rcol, gcol, bcol))

cm_linear = matplotlib.colors.LinearSegmentedColormap.from_list("PaulT_linear", cols)

# Linear colormap (rainbow)
cols = []
# cols = [(0,0,0)]
for x in N.linspace(0,1, 256):
	rcol = (0.472-0.567*x+4.05*x**2)/(1.+8.72*x-19.17*x**2+14.1*x**3)
	gcol = 0.108932-1.22635*x+27.284*x**2-98.577*x**3+163.3*x**4-131.395*x**5+40.634*x**6
	bcol = 1./(1.97+3.54*x-68.5*x**2+243*x**3-297*x**4+125*x**5)
	cols.append((rcol, gcol, bcol))

# cols.append((1,1,1))
cm_rainbow = matplotlib.colors.LinearSegmentedColormap.from_list("PaulT_rainbow", cols)

if __name__ == '__main__':
	# Plot examples
	import matplotlib.pyplot as plt
	plt.ion()
	tmpim = N.arange(256).reshape(1,-1)
	plt.close()
	plt.title("www.sron.nl/~pault variation around zero colormap")
	plt.imshow(tmpim, cmap=plt.get_cmap(cm_plusmin), aspect='auto')
	plt.savefig("matplotlib-ref-plusmin.pdf")

	plt.close()
	plt.title("www.sron.nl/~pault linear colormap")
	plt.imshow(tmpim, cmap=plt.get_cmap(cm_linear), aspect='auto')
	plt.savefig("matplotlib-ref-linear.pdf")

	plt.close()
	plt.title("www.sron.nl/~pault rainbow colormap")
	plt.imshow(tmpim, cmap=plt.get_cmap(cm_rainbow), aspect='auto')
	plt.savefig("matplotlib-ref-rainbow.pdf")


# EOF
