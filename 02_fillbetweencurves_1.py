#!/usr/bin/python
#####################################################################
# Author:		Alberto Albz Marocchino
# Date:			19-02-2017
# Purpose:      comparison between python and R :: Shadows graphs
# Source:       python
#####################################################################

import numpy as np
import pylab as pyl
from scipy import stats
import numpy.polynomial.polynomial as poly

#--- generate random samples---#
N=100
x=np.linspace(0,10,N)
y=x
y_up = y + 1 + np.random.normal(0, .3, N)
y_do = y - 1 + np.random.normal(0, .3, N)

#--- plot the two samples ---#
%matplotlib inline
pyl.fill_between(x, y_do, y_up, color='lightgray')
pyl.plot(x,y,'-',color='magenta')
