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
y1, y2, y3 =x, 1.1*x, 0.8*x
y_up1, y_up2, y_up3 = y1 + 1 + np.random.normal(0, .3, N), y2 + 1 + np.random.normal(0, .3, N), y3 + 1 + np.random.normal(0, .3, N)
y_do1, y_do2, y_do3 = y1 - 1 + np.random.normal(0, .3, N), y2 - 1 + np.random.normal(0, .3, N), y3 - 1 + np.random.normal(0, .3, N)

#--- plot the two samples ---#
%matplotlib inline
pyl.fill_between(x, y_do1, y_up1, color='blue', alpha=0.35)
pyl.fill_between(x, y_do2, y_up2, color='grey', alpha=0.35)
pyl.fill_between(x, y_do3, y_up3, color='green', alpha=0.35)
pyl.plot(x,y1,'-',color='blue')
pyl.plot(x,y2,'-',color='black')
pyl.plot(x,y3,'-',color='green')
