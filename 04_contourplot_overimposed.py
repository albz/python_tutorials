#!/usr/bin/python
######################################################################
# Name:         im_300.py
# Author:       A. Marocchino
# Date:			2017-08-30
# Purpose:      plot Lu model solutions for PW-shock-injection
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


# --- *** ColorMaps *** --- #
ne_cmap=truncate_colormap(plt.get_cmap('Greys'), minval=0.10, maxval=0.75, n=100)
bunch_cmap = plt.get_cmap('plasma')
# --- #


# --- paths --- #
path = os.path.join(os.getcwd(),'04_data')


# --- data+flipping --- #
z_mesh  = np.loadtxt(os.path.join(path,'z_mesh.dat'))
r_mesh  = np.loadtxt(os.path.join(path,'r_mesh.dat'))
rho_b   = np.loadtxt(os.path.join(path,'bunch_density.dat'))
rho_bck = np.loadtxt(os.path.join(path,'background_density.dat'))

z_mesh=z_mesh-220; rho_bck=np.fliplr(rho_bck); rho_b=np.fliplr(rho_b);

# --- *** --- #
fig = pyl.figure(1)
fig.set_size_inches(3.25, 3.0, forward=True)
ax1  = pyl.subplot(111)

rho_min_bck=0.1
rho_max_bck=5.0
rho_min_b=0.1
rho_max_b=40.0
extent = (np.min(z_mesh),np.max(z_mesh),np.min(r_mesh),np.max(r_mesh))
im = ax1.imshow(rho_bck,interpolation = 'bicubic', cmap = ne_cmap, aspect = 'auto' , extent=extent, clim=[rho_min_bck, rho_max_bck], alpha=0.999)
im2 = Imshow(rho_b,interpolation = 'bicubic', cmap =bunch_cmap, aspect = 'auto' , extent=extent, clim=[rho_min_b, rho_max_b], tvmin=0.1, tvmax=0.7,vmin=rho_min_b,vmax=rho_max_b)

#--- plot density ---#
ax1.plot(z_mesh,rho_b[248,:])

ax1.set_xlim(-250,150)
ax1.set_ylim(-60,60)


ax1.set_ylabel('R ($\mu$m)', labelpad=-12)
ax1.set_xlabel('$\zeta$ ($\mu$m)', labelpad=0)

cbaxes_bck = fig.add_axes([0.82, 0.57, 0.035, 0.38])
cbar_bck = plt.colorbar(im,ax = ax1,cmap=ne_cmap,cax = cbaxes_bck,format = '%1.1f' ,ticks = [0.,2.5,5.])#,aspect = 5,fraction = 0.07)
cbar_bck.ax.tick_params(labelsize=7)

ax1.text(135, 60, r'$n/10^{16}$', fontsize=8)
ax1.text(-45,-15, 'Driver', fontsize=10, color='black')
ax1.text(-215,-23, 'trailing\nbunch', fontsize=10, color='black')

cbaxes_b = fig.add_axes([0.82, 0.14, 0.035, 0.38])
cbar_b = plt.colorbar(im2,ax = ax1,cmap=bunch_cmap,cax = cbaxes_b,format = '%1.1f' ,ticks = [0.1,20.0,40.0]) #aspect = 5,fraction = 0.07)
cbar_b.ax.tick_params(labelsize=7)


pyl.subplots_adjust(bottom=0.14,right=0.82)
# pyl.savefig('./im_04.pdf', format='pdf', dpi=500)
plt.show()
