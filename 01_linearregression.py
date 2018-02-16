#!/usr/bin/python
#####################################################################
# Author:		Alberto Albz Marocchino
# Date:			19-02-2017
# Purpose:     comparison between python and R :: Linear Regression
# Source:       python
#####################################################################

import numpy as np
import pylab as pyl
from scipy import stats
import numpy.polynomial.polynomial as poly

#--- generate sample---#
x=np.linspace(0,10,100)
y_therory=x
y=y_therory+np.random.normal(0, 1, 100)

#---linear interpolation---#
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
coefficients = poly.polyfit(x, y, 1)
y_reconstructed = poly.polyval(x, coefficients)


%matplotlib inline
pyl.plot(x,y,'.',label='scattered data')
pyl.plot(x,y_therory,'x',label='original data')
pyl.plot(x,y_reconstructed,'-',label='fit',lw=3)
pyl.legend()
