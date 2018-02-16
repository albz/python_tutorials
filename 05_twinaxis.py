#!/usr/bin/python
######################################################################
# Name:         im_313.py
# Author:       A. Marocchino and E. Brentegani
# Date:			2018-01-23
# Purpose:      plot active plasma lens bending-B-field plus particles
# Source:       python
#####################################################################

### loading shell commands
import os, os.path, glob, sys, shutil
import time, datetime
import scipy
import numpy as np
import matplotlib
import matplotlib.patches as patches
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import pylab as pyl
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
plt.style.use(os.path.join('.','utilities','plot_style_ppth.mplstyle'))
sys.path.append(os.path.join('.','utilities'))
from plot_utilities import *
### --- ###


# --- path --- #
path = os.path.join(os.getcwd(),'05_data')

# --- Bfield conversion (from dimensionless to milli-Tesla)--- #
Bconst = 96.*np.sqrt(1e16)/3e8 * 1e3

# --- *** --- #
fig = pyl.figure(1)
fig.set_size_inches(3.25, 3.0, forward=True)
ax1  = pyl.subplot(111)

#load and plot data :: 175 ns
r_mesh  = np.loadtxt(os.path.join(path,'r_mesh_01.dat'))
Bphi_ex = np.loadtxt(os.path.join(path,'Bex_01.dat'))
ax1.plot(r_mesh,Bphi_ex * Bconst, '-', color='blue',  lw=1, label='175ns - 10A')

#load and plot data :: 250 ns
r_mesh  = np.loadtxt(os.path.join(path,'r_mesh_02.dat'))
Bphi_ex = np.loadtxt(os.path.join(path,'Bex_02.dat'))
ax1.plot(r_mesh,Bphi_ex[:] * Bconst, '-', color='red',  lw=1, label='250ns - 45A')

#load and plot data :: 325 ns
r_mesh  = np.loadtxt(os.path.join(path,'r_mesh_03.dat'))
Bphi_ex = np.loadtxt(os.path.join(path,'Bex_03.dat'))
ax1.plot(r_mesh,Bphi_ex[:] * Bconst, '--', color='black',  lw=1, label='325ns - 95A')

#Limits
ax1.set_xlim(-500,500)
ax1.set_ylim(-40,40)

#Ticks
ax1.xaxis.set_major_locator(MultipleLocator(250))
ax1.xaxis.set_minor_locator(MultipleLocator(50))
ax1.yaxis.set_major_locator(MultipleLocator(20))
ax1.yaxis.set_minor_locator(MultipleLocator(5))

#Labels
ax1.set_ylabel('B-field (mT)', labelpad=-4)
ax1.set_xlabel(r'x ($\mu$m)', labelpad=1)

#Legend
ax1.legend(loc=1, ncol=1, prop={'size':7.5}, frameon=False)

#--- --- ----#
ax2 = ax1.twinx()
#load and plot hystogram :: 325 ns
[x,y,z,px,py,pz,macro_particle_number] = np.transpose(np.loadtxt(os.path.join(path,'phase_space.dat')))
hist, bin_edges = np.histogram(x, bins=100, range=(-500,500), density=False)
ax2.bar(bin_edges[0:-1],hist*1.6e-19*1e12*macro_particle_number[0],width=10,facecolor='g')

#Limits
ax2.set_xlim(-500,500)
ax2.set_ylim(0.,2.)

#Labels
ax2.set_ylabel('Slice charge (pC)', labelpad=1)

#Ticks
ax2.yaxis.set_major_locator(MultipleLocator(.5))
ax2.yaxis.set_minor_locator(MultipleLocator(.1))

#--- ordering ---#
ax1.set_zorder(ax2.get_zorder()+1)
ax1.patch.set_visible(False)

pyl.subplots_adjust(top=0.92,left=0.15,right=0.84)
# pyl.savefig('./im_05.png', format='png', dpi=1200)
plt.show()
