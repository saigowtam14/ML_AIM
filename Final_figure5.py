# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 10:05:26 2023

@author: SAI GOWTAM VALLURI
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import warnings
from datetime import timedelta
warnings.filterwarnings("ignore")
# =============================================================================
with open('20170907_2201.pkl','rb') as f:
    mlx, mly, mlz, dmx, dmy, dmz, wx, wy, wz, sdx, sdy, sdz, cpdmsp,\
        weimertm,tmpp,vmlats,vmlons = pickle.load(f)
f.close()
mlx,mly,mlz = np.load('ml201709072202.npy')
# plt.rcParams['axes.grid'] = False
# plt.rcParams.update({'font.size': 16})
# plt.rcParams.update({'font.weight': 'bold'})
fig,axes = plt.subplots(figsize=(15,20),dpi=600)
plt.subplots_adjust(wspace=0)
plt.subplots_adjust(left=0.0, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.3)
   
cpcp =  np.round(np.max(mlz)-np.min(mlz))
cpcp = np.round(cpcp)
vmin = -75
vmax = 75
cmap = 'seismic'
norm = colors.Normalize(vmin=vmin, vmax=vmax)
ax1 = plt.subplot(3,3,1,projection='polar')
qcs1 = ax1.pcolormesh(mly, mlx, mlz, norm=norm, cmap = cmap)
cs = ax1.contour(mly, mlx, mlz, colors = 'k',levels=range(-60,61,10))
plt.clabel(cs, fmt = '%2.1d', colors = 'k', fontsize=6)
ax1.set_theta_zero_location("S")
ax1.set_ylim(0, 40)
ax1.xaxis.grid(linestyle='--', linewidth=0.8)
ax1.yaxis.grid(linestyle='--', linewidth=0.8)
# cbar1 = fig.colorbar(qcs1, ax=ax1,shrink=0.4,pad=0.1)
# cbar1.set_label('Potential (kV)',fontweight="bold")
xtickpos = ax1.get_xticks()
xticks = ['00 MLT', '03','06','09','12','15','18','21']
ax1.set_xticks(xtickpos,xticks)
ytickpos = [5,10,15,20,25,30,35,40]
yticks = ['N', '80$^\circ$','','70$^\circ$','','60$^\circ$','','50$^\circ$']
ax1.set_yticks(ytickpos,yticks)
qcs1 = ax1.quiver(dmy, dmx, dmy, dmx+dmz,scale=15000,color='m')
plt.title('(a) ML-AIM',fontweight="bold", fontsize = 16)
cpcp1 = '$\u03A6_{PC}$ = '+str(cpcp)+' kV'
ax1.text(0.25, -0.32, cpcp1, transform=ax1.transAxes, fontsize = 16)
tmpp = pd.to_datetime(tmpp) + timedelta(minutes=1)
ax1.text(0.25, -0.2, tmpp, transform=ax1.transAxes, fontsize = 16, color='m')
cpcp1 = 'DMSP $\u03A6_{PC}$ = '+str(cpdmsp)+' kV'
ax1.text(0.25, -0.26, cpcp1, transform=ax1.transAxes, fontsize = 16, color='m')


wxc = wx[wx<=28]
wyc = wy
wzc = wz[0:29,:]

cpcp =  np.round(np.max(wz)-np.min(wz))
ax1 = plt.subplot(3,3,2,projection='polar')
qcs1 = ax1.pcolormesh(wy, wx, wz,norm=norm, cmap = cmap)
cs = ax1.contour(wyc, wxc, wzc, colors = 'k',levels=range(-60,61,10))
plt.clabel(cs, fmt = '%2.1d', colors = 'k', fontsize=6)
cpcp1 = 'Cross Polar Cap Potential = '+str(cpcp)+' kV'
ax1.set_theta_zero_location("S")
ax1.set_ylim(0, 40)
# ax1.set_title(cpcp1,fontweight="bold")
ax1.xaxis.grid(linestyle='--', linewidth=0.8)
ax1.yaxis.grid(linestyle='--', linewidth=0.8)
# cbar1 = fig.colorbar(qcs1, ax=ax1,shrink=0.4,pad=0.1)
# cbar1.set_label('Potential (kV)',fontweight="bold")
xtickpos = ax1.get_xticks()
xticks = ['00 MLT', '03','06','09','12','15','18','21']
ax1.set_xticks(xtickpos,xticks)
ytickpos = [5,10,15,20,25,30,35,40]
yticks = ['N', '80$^\circ$','','70$^\circ$','','60$^\circ$','','50$^\circ$']
ax1.set_yticks(ytickpos,yticks)
qcs1 = ax1.quiver(dmy, dmx, dmy, dmx+dmz,scale=15000,color='m')
plt.title('(b) Weimer 2005',fontweight="bold", fontsize = 16)
cpcp1 = '$\u03A6_{PC}$ = '+str(cpcp)+' kV'
ax1.text(0.25, -0.2, cpcp1, transform=ax1.transAxes, fontsize = 16)




# cpcp = np.round(np.max(sdz)-np.min(sdz))
cpcp = 91.7 #SuperDARN 
norm = colors.Normalize(vmin=vmin, vmax=vmax)
ax1 = plt.subplot(3,3,3,projection='polar')
qcs1 = ax1.pcolormesh(sdy, sdx, sdz, norm=norm, cmap=cmap)
cs = ax1.contour(sdy, sdx, sdz, colors = 'k',levels=range(-60,61,10))
plt.clabel(cs, fmt = '%2.1d', colors = 'k', fontsize=7)
ax1.set_theta_zero_location("S")
ax1.set_theta_zero_location("S")
ax1.set_ylim(0, 40)
ax1.scatter(vmlons,90-vmlats,s=3,color='g')
ax1.grid(linestyle='--', linewidth=0.8)
ax1.grid(linestyle='--', linewidth=0.8)
# cbar1 = fig.colorbar(qcs1, ax=ax1,shrink=0.4,pad=0.1)
# cbar1.set_label('Potential (kV)',fontweight="bold")
xtickpos = ax1.get_xticks()
xticks = ['00 MLT', '03','06','09','12','15','18','21']
ax1.set_xticks(xtickpos,xticks)
ytickpos = [5,10,15,20,25,30,35,40]
yticks = ['N', '80$^\circ$','','70$^\circ$','','60$^\circ$','','50$^\circ$']
ax1.set_yticks(ytickpos,yticks)
qcs1 = ax1.quiver(dmy, dmx, dmy, dmx+dmz,scale=15000,color='m')
plt.title('(c) SuperDARN',fontweight="bold", fontsize = 16)
cpcp1 = '$\u03A6_{PC}$ = '+str(cpcp)+' kV'
ax1.text(0.25, -0.2, cpcp1, transform=ax1.transAxes, fontsize = 16)



with open('20170907_2342.pkl','rb') as f:  # Python 3: open(..., 'rb')
    mlx, mly, mlz, dmx, dmy, dmz, wx, wy, wz, sdx, sdy, sdz, cpdmsp,\
        weimertm,tmpp,vmlats,vmlons = pickle.load(f)
f.close()

mlx,mly,mlz = np.load('ml201709072342.npy')
  
cpcp =  np.round(np.max(mlz)-np.min(mlz))
cpcp = np.round(cpcp)
vmin = -75
vmax = 75
cmap = 'seismic'
norm = colors.Normalize(vmin=vmin, vmax=vmax)
ax1 = plt.subplot(3,3,4,projection='polar')
# ax1.set_position([0.26, 0.67, 0.2, 0.2])
qcs1 = ax1.pcolormesh(mly, mlx, mlz, norm=norm, cmap = cmap)
cs = ax1.contour(mly, mlx, mlz, colors = 'k',levels=range(-60,61,10))
plt.clabel(cs, fmt = '%2.1d', colors = 'k', fontsize=6)
ax1.set_theta_zero_location("S")
ax1.set_ylim(0, 40)
ax1.grid(linestyle='--', linewidth=0.8)
ax1.grid(linestyle='--', linewidth=0.8)
# cbar1 = fig.colorbar(qcs1, ax=ax1,shrink=0.4,pad=0.1)
# cbar1.set_label('Potential (kV)',fontweight="bold")
xtickpos = ax1.get_xticks()
xticks = ['00 MLT', '03','06','09','12','15','18','21']
ax1.set_xticks(xtickpos,xticks)
ytickpos = [5,10,15,20,25,30,35,40]
yticks = ['N', '80$^\circ$','','70$^\circ$','','60$^\circ$','','50$^\circ$']
ax1.set_yticks(ytickpos,yticks)
qcs1 = ax1.quiver(dmy, dmx, dmy, dmx+dmz,scale=15000,color='m')
plt.title('(d) ML-AIM',fontweight="bold", fontsize = 16)
cpcp1 = '$\u03A6_{PC}$ = '+str(cpcp)+' kV'
ax1.text(0.25, -0.32, cpcp1, transform=ax1.transAxes, fontsize = 16)
tmpp = pd.to_datetime(tmpp) + timedelta(minutes=1)
ax1.text(0.25, -0.2, tmpp, transform=ax1.transAxes, fontsize = 16, color='m')
cpcp1 = 'DMSP $\u03A6_{PC}$ = '+str(cpdmsp)+' kV'
ax1.text(0.25, -0.26, cpcp1, transform=ax1.transAxes, fontsize = 16, color='m')


wxc = wx[wx<=28]
wyc = wy
wzc = wz[0:29,:]

cpcp =  np.round(np.max(wz)-np.min(wz))
ax1 = plt.subplot(3,3,5,projection='polar')
qcs1 = ax1.pcolormesh(wy, wx, wz,norm=norm, cmap = cmap)
cs = ax1.contour(wyc, wxc, wzc, colors = 'k',levels=range(-60,61,10))
plt.clabel(cs, fmt = '%2.1d', colors = 'k', fontsize=6)
cpcp1 = 'Cross Polar Cap Potential = '+str(cpcp)+' kV'
ax1.set_theta_zero_location("S")
ax1.set_ylim(0, 40)
# ax1.set_title(cpcp1,fontweight="bold")
ax1.grid(linestyle='--', linewidth=0.8)
ax1.grid(linestyle='--', linewidth=0.8)
# cbar1 = fig.colorbar(qcs1, ax=ax1,shrink=0.4,pad=0.1)
# cbar1.set_label('Potential (kV)',fontweight="bold")
xtickpos = ax1.get_xticks()
xticks = ['00 MLT', '03','06','09','12','15','18','21']
ax1.set_xticks(xtickpos,xticks)
ytickpos = [5,10,15,20,25,30,35,40]
yticks = ['N', '80$^\circ$','','70$^\circ$','','60$^\circ$','','50$^\circ$']
ax1.set_yticks(ytickpos,yticks)
qcs1 = ax1.quiver(dmy, dmx, dmy, dmx+dmz,scale=15000,color='m')
plt.title('(e) Weimer 2005',fontweight="bold", fontsize = 16)
cpcp1 = '$\u03A6_{PC}$ = '+str(cpcp)+' kV'
ax1.text(0.25, -0.2, cpcp1, transform=ax1.transAxes, fontsize = 16)




# cpcp = np.round(np.max(sdz)-np.min(sdz))
cpcp = 89.4 #SuperDARN 
norm = colors.Normalize(vmin=vmin, vmax=vmax)
ax1 = plt.subplot(3,3,6,projection='polar')
qcs1 = ax1.pcolormesh(sdy, sdx, sdz, norm=norm, cmap=cmap)
cs = ax1.contour(sdy, sdx, sdz, colors = 'k',levels=range(-60,61,10))
plt.clabel(cs, fmt = '%2.1d', colors = 'k', fontsize=7)
ax1.set_theta_zero_location("S")
ax1.set_theta_zero_location("S")
ax1.set_ylim(0, 40)
ax1.scatter(vmlons,90-vmlats,s=3,color='g')
ax1.grid(linestyle='--', linewidth=0.8)
ax1.grid(linestyle='--', linewidth=0.8)
# cbar1 = fig.colorbar(qcs1, ax=ax1,shrink=0.4,pad=0.1)
# cbar1.set_label('Potential (kV)',fontweight="bold")
xtickpos = ax1.get_xticks()
xticks = ['00 MLT', '03','06','09','12','15','18','21']
ax1.set_xticks(xtickpos,xticks)
ytickpos = [5,10,15,20,25,30,35,40]
yticks = ['N', '80$^\circ$','','70$^\circ$','','60$^\circ$','','50$^\circ$']
ax1.set_yticks(ytickpos,yticks)
qcs1 = ax1.quiver(dmy, dmx, dmy, dmx+dmz,scale=15000,color='m')
plt.title('(f) SuperDARN',fontweight="bold", fontsize = 16)
cpcp1 = '$\u03A6_{PC}$ = '+str(cpcp)+' kV'
ax1.text(0.25, -0.2, cpcp1, transform=ax1.transAxes, fontsize = 16)
# plt.tight_layout()




with open('20170908_1503.pkl','rb') as f:  # Python 3: open(..., 'rb')
    mlx, mly, mlz, dmx, dmy, dmz, wx, wy, wz, sdx, sdy, sdz, cpdmsp,\
        weimertm,tmpp,vmlats,vmlons = pickle.load(f)
f.close()
mlx,mly,mlz = np.load('ml201709081504.npy')
cpcp =  np.round(np.max(mlz)-np.min(mlz))
cpcp = np.round(cpcp)
vmin = -75
vmax = 75
cmap = 'seismic'
norm = colors.Normalize(vmin=vmin, vmax=vmax)
ax1 = plt.subplot(3,3,7,projection='polar')
# ax1.set_position([0.26, 0.67, 0.2, 0.2])
qcs1 = ax1.pcolormesh(mly, mlx, mlz, norm=norm, cmap = cmap)
cs = ax1.contour(mly, mlx, mlz, colors = 'k',levels=range(-60,61,10))
plt.clabel(cs, fmt = '%2.1d', colors = 'k', fontsize=6)
ax1.set_theta_zero_location("S")
ax1.set_ylim(0, 40)
ax1.xaxis.grid(linestyle='--', linewidth=0.8)
ax1.yaxis.grid(linestyle='--', linewidth=0.8)
# cbar1.set_label('Potential (kV)',fontweight="bold")
xtickpos = ax1.get_xticks()
xticks = ['00 MLT', '03','06','09','12','15','18','21']
ax1.set_xticks(xtickpos,xticks)
ytickpos = [5,10,15,20,25,30,35,40]
yticks = ['N', '80$^\circ$','','70$^\circ$','','60$^\circ$','','50$^\circ$']
ax1.set_yticks(ytickpos,yticks)
qcs1 = ax1.quiver(dmy, dmx, dmy, dmx+dmz,scale=15000,color='m')
plt.title('(g) ML-AIM',fontweight="bold", fontsize = 16)
cpcp1 = '$\u03A6_{PC}$ = '+str(cpcp)+' kV'
ax1.text(0.25, -0.32, cpcp1, transform=ax1.transAxes, fontsize = 16)
tmpp = pd.to_datetime(tmpp) + timedelta(minutes=1)
ax1.text(0.25, -0.2, tmpp, transform=ax1.transAxes, fontsize = 16, color='m')
cpcp1 = 'DMSP $\u03A6_{PC}$ = '+str(cpdmsp)+' kV'
ax1.text(0.25, -0.26, cpcp1, transform=ax1.transAxes, fontsize = 16, color='m')


wxc = wx[wx<=24]
wyc = wy
wzc = wz[0:25,:]

cpcp =  np.round(np.max(wz)-np.min(wz))
ax1 = plt.subplot(3,3,8,projection='polar')
qcs1 = ax1.pcolormesh(wy, wx, wz,norm=norm, cmap = cmap)
cs = ax1.contour(wyc, wxc, wzc, colors = 'k',levels=range(-60,61,10))
plt.clabel(cs, fmt = '%2.1d', colors = 'k', fontsize=6)
cpcp1 = 'Cross Polar Cap Potential = '+str(cpcp)+' kV'
ax1.set_theta_zero_location("S")
ax1.set_ylim(0, 40)
# ax1.set_title(cpcp1,fontweight="bold")
ax1.xaxis.grid(linestyle='--', linewidth=0.8)
ax1.yaxis.grid(linestyle='--', linewidth=0.8)
# cbar1 = fig.colorbar(qcs1, ax=ax1,shrink=0.4,pad=0.1)
# cbar1.set_label('Potential (kV)',fontweight="bold")
xtickpos = ax1.get_xticks()
xticks = ['00 MLT', '03','06','09','12','15','18','21']
ax1.set_xticks(xtickpos,xticks)
ytickpos = [5,10,15,20,25,30,35,40]
yticks = ['N', '80$^\circ$','','70$^\circ$','','60$^\circ$','','50$^\circ$']
ax1.set_yticks(ytickpos,yticks)
qcs1 = ax1.quiver(dmy, dmx, dmy, dmx+dmz,scale=15000,color='m')
plt.title('(h) Weimer 2005',fontweight="bold", fontsize = 16)
cpcp1 = '$\u03A6_{PC}$ = '+str(cpcp)+' kV'
ax1.text(0.25, -0.2, cpcp1, transform=ax1.transAxes, fontsize = 16)




# cpcp = np.round(np.max(sdz)-np.min(sdz))
cpcp = 40.0 #SuperDARN 
norm = colors.Normalize(vmin=vmin, vmax=vmax)
ax1 = plt.subplot(3,3,9,projection='polar')
qcs1 = ax1.pcolormesh(sdy, sdx, sdz, norm=norm, cmap=cmap)
cs = ax1.contour(sdy, sdx, sdz, colors = 'k',levels=range(-60,61,10))
plt.clabel(cs, fmt = '%2.1d', colors = 'k', fontsize=7)
ax1.set_theta_zero_location("S")
ax1.set_theta_zero_location("S")
ax1.set_ylim(0, 40)
ax1.scatter(vmlons,90-vmlats,s=3,color='g')
ax1.xaxis.grid(linestyle='--', linewidth=0.8)
ax1.yaxis.grid(linestyle='--', linewidth=0.8)
# cbar1 = fig.colorbar(qcs1, ax=ax1,shrink=0.4,pad=0.1)
# cbar1.set_label('Potential (kV)',fontweight="bold")
xtickpos = ax1.get_xticks()
xticks = ['00 MLT', '03','06','09','12','15','18','21']
ax1.set_xticks(xtickpos,xticks)
ytickpos = [5,10,15,20,25,30,35,40]
yticks = ['N', '80$^\circ$','','70$^\circ$','','60$^\circ$','','50$^\circ$']
ax1.set_yticks(ytickpos,yticks)

plt.title('(i) SuperDARN',fontweight="bold", fontsize = 16)
cax = fig.add_axes([.95,0.3,0.01,0.4])
cbar1 = fig.colorbar(qcs1,cax=cax)
cbar1.set_label('Potential (kV)',fontweight="bold")
qcs1 = ax1.quiver(dmy, dmx, dmy, dmx+dmz,scale=15000,color='m')
cpcp1 = '$\u03A6_{PC}$ = '+str(cpcp)+' kV'
ax1.text(0.25, -0.2, cpcp1, transform=ax1.transAxes, fontsize = 16)

plt.savefig('Figure5.pdf',format='pdf')
# plt.tight_layout()
plt.show()

