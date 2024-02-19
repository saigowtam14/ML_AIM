# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 09:42:30 2023

@author: SAI GOWTAM VALLURI
"""

import pyIGRF, pyproj, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import warnings
warnings.filterwarnings("ignore")


xfac,yfac,amp_pred,xsigp,ysigp,sigp,xsigh,ysigh,sigh,xpot,ypot,pot,xjh,yjh,zjh,xj,yj,z1,z2 = np.load('Figure2_data.npy')

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.weight': 'bold'})
fig,axes = plt.subplots(figsize=(15,18),dpi=600)

vmin = -1;
vmax = 1;
norm = colors.Normalize(vmin=vmin, vmax=vmax)
cmap = 'seismic'
ax1 = plt.subplot(3,2,1,projection='polar')
qcs1 = ax1.pcolormesh(yfac, xfac, amp_pred, norm=norm, cmap=cmap)
ax1.set_theta_zero_location("S")
ax1.set_ylim(0, 40)
ax1.xaxis.grid(linestyle='--', linewidth=0.8)
ax1.yaxis.grid(linestyle='--', linewidth=0.8)
# ax1.set_title('(a) Field Aligned Currents '+ fname,fontweight="bold")
ax1.set_title('(a) Field Aligned Currents',fontweight="bold")
cbar1 = fig.colorbar(qcs1, ax=ax1,shrink=0.5,pad=0.1)
cbar1.set_label('($\u03BC A/m^2$)',fontweight="bold")
xtickpos = ax1.get_xticks()
xticks = ['00', '03','06','09','12','15','18','21']
ax1.set_xticks(xtickpos,xticks)
ytickpos = [5., 10., 15., 20., 25., 30., 35., 40.]
yticks = ['N', '80$^\circ$','','70$^\circ$','','60$^\circ$','','50$^\circ$']
ax1.set_yticks(ytickpos,yticks)


vmin = 0;
vmax = 30;
norm = colors.Normalize(vmin=vmin, vmax=vmax)
ax1 = plt.subplot(3,2,3,projection='polar')
qcs1 = ax1.pcolormesh(ysigp, xsigp, sigp, norm=norm, cmap='Reds')
ax1.set_theta_zero_location("S")
ax1.set_ylim(0, 40)
ax1.xaxis.grid(linestyle='--', linewidth=0.8)
ax1.yaxis.grid(linestyle='--', linewidth=0.8)
ax1.set_title('(b) Pedersen conductance',fontweight="bold")
cbar1 = fig.colorbar(qcs1, ax=ax1,shrink=0.5,pad=0.1)
cbar1.set_label('(S)',fontweight="bold")
xtickpos = ax1.get_xticks()
xticks = ['00', '03','06','09','12','15','18','21']
ax1.set_xticks(xtickpos,xticks)
ytickpos = [5., 10., 15., 20., 25., 30., 35., 40.]
yticks = ['N', '80$^\circ$','','70$^\circ$','','60$^\circ$','','50$^\circ$']
ax1.set_yticks(ytickpos,yticks)



ax1 = plt.subplot(3,2,5,projection='polar')
norm = colors.Normalize(vmin=vmin, vmax=vmax)
qcs1 = ax1.pcolormesh(ysigh, xsigh, sigh, norm=norm, cmap='Reds')
ax1.set_theta_zero_location("S")
ax1.set_ylim(0, 40)
ax1.set_title('(c) Hall conductance',fontweight="bold")
ax1.xaxis.grid(linestyle='--', linewidth=0.8)
ax1.yaxis.grid(linestyle='--', linewidth=0.8)
cbar1 = fig.colorbar(qcs1, ax=ax1,shrink=0.5,pad=0.1)
cbar1.set_label('(S)',fontweight="bold")
xtickpos = ax1.get_xticks()
xticks = ['00', '03','06','09','12','15','18','21']
ax1.set_xticks(xtickpos,xticks)
ytickpos = [5., 10., 15., 20., 25., 30., 35., 40.]
yticks = ['N', '80$^\circ$','','70$^\circ$','','60$^\circ$','','50$^\circ$']
ax1.set_yticks(ytickpos,yticks)



vmin = -50;
vmax = 50;
norm = colors.Normalize(vmin=vmin, vmax=vmax)
ax1 = plt.subplot(3,2,2,projection='polar')
qcs1 = ax1.pcolormesh(ypot, xpot, pot, norm=norm, cmap=cmap)
cs = ax1.contour(ypot, xpot, pot, colors = 'k',levels=range(-40,41,10))
plt.clabel(cs, fmt = '%2.1d', colors = 'k')
cpcp = 63.0
cpcp1 = 'Cross Polar Cap Potential = '+str(cpcp)+' kV'
ax1.set_theta_zero_location("S")
ax1.set_ylim(0, 40)
ax1.set_title('(d) Potential',fontweight="bold")
ax1.xaxis.grid(linestyle='--', linewidth=0.8)
ax1.yaxis.grid(linestyle='--', linewidth=0.8)
cbar1 = fig.colorbar(qcs1, ax=ax1,shrink=0.5,pad=0.1)
cbar1.set_label('(kV)',fontweight="bold")
xtickpos = ax1.get_xticks()
xticks = ['00 MLT', '03','06','09','12','15','18','21']
ax1.set_xticks(xtickpos,xticks)
ytickpos = [5., 10., 15., 20., 25., 30., 35., 40.]
yticks = ['N', '80$^\circ$','','70$^\circ$','','60$^\circ$','','50$^\circ$']
ax1.set_yticks(ytickpos,yticks)



file_name = 'dms_20130514_17s1.001.hdf5.txt'
start_tm = '20130514 12:00:00'
end_tm = '20130514 13:10:00'
start_tm = pd.to_datetime(start_tm)
end_tm = pd.to_datetime(end_tm)
tmpp = pd.to_datetime('20130514 12:44:00')

cols = ['year','month','day','hour','minute','second','RECNO','RECNO1','KINDAT','UT1_UNIX','UT2_UNIX',\
        'gdlat','glon','gdalt','SAT_ID','mlt','mlat','MLONG','NE','hor_ion_v','vert_ion_v','BD',\
            'B_FORWARD','B_PERP', 'DIFF_BD','DIFF_B_FOR','DIFF_B_PERP']
dmsp1 = pd.read_csv(file_name,names=cols,skiprows=1,delim_whitespace=True,low_memory=False)
dmsp1['timestamps'] = pd.to_datetime(dmsp1[['month','day','year','hour','minute']]);
dmsp1 = dmsp1[dmsp1['mlat'].between(40,90)]
dmsp1 = dmsp1.reset_index()
dmsp1 = dmsp1[dmsp1['hor_ion_v'].between(-3000,3000)]
dmsp1 = dmsp1.reset_index()

intind = (dmsp1['timestamps']>=start_tm) & (dmsp1['timestamps']<=end_tm)
dmsp1 = dmsp1.loc[intind]
dmsp1.reset_index(inplace=True,drop=True)


lat = np.array(dmsp1['gdlat'])
lon = np.array(dmsp1['glon'])
alt = np.array(dmsp1['gdalt'])
date = np.repeat(2013,np.size(alt))
igrf = pd.DataFrame([])
for i1 in range(0,np.size(alt)):
    ig =  pyIGRF.igrf_value(lat[i1], lon[i1], alt[i1], date[i1])
    ig = np.transpose(ig)
    tmpig = pd.DataFrame(ig)
    igrf = pd.concat([igrf,tmpig])

igrf = np.reshape(np.array(igrf),[int(np.size(igrf)/7),7])

Vy = np.array(dmsp1['hor_ion_v'])
Vz = np.array(dmsp1['vert_ion_v'])
Bx = igrf[:,3]*(10**-9)
By = igrf[:,4]*(10**-9)
Bz = igrf[:,5]*(10**-9)
dmsp1['Bx'] = Bx
dmsp1['By'] = By
dmsp1['Bz'] = Bz
 
Ex = -Vy*Bz + Vz*By
dmsp1['Ex'] = Ex 
Exd1 = Ex


x1 = 90-dmsp1['mlat'];
y1 = dmsp1['mlt']*15;
y1 = np.deg2rad(y1);
z1 = dmsp1['hor_ion_v'];
qcs1 = ax1.quiver(y1, x1, y1, x1+z1,scale=15000,color='m')

tmm = pd.to_datetime(tmpp)
title2 = str(tmm.year)+ "{:02d}".format(tmm.month) +\
    "{:02d}".format(tmm.day) +' '+"{:02d}".format(tmm.hour) + \
        ':'+ "{:02d}".format(tmm.minute) + 'UT'
# plt.title('(b) Weimer model')
cpcp1 = '$\u03A6_{PC}$ = '+str(cpcp)+' kV'
ax1.text(1, 1, cpcp1, transform=ax1.transAxes)


ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')


for i2 in range(0,np.size(alt)-1):
    a = pyproj.transform(lla, ecef, lon[i2], lat[i2], alt[i2], radians=False)
    b = pyproj.transform(lla, ecef, lon[i2+1], lat[i2+1], alt[i2+1], radians=False)
    d = math.dist(b, a)
    Exd1[i2] = Ex[i2]*(d)

Exd1[Exd1>750] = np.nan
Exd1[Exd1<-750] = np.nan

dmsp1['Ex*dl_igrf'] = Exd1

tmp1 = np.array(dmsp1['Ex*dl_igrf'].cumsum())

dl_cor = []
for i2 in range(0,np.size(alt)):
    a1 = pyproj.transform(lla, ecef, lon[0], lat[0], alt[0], radians=False)
    b1 = pyproj.transform(lla, ecef, lon[-1], lat[-1], alt[-1], radians=False)
    d1 = math.dist(b1, a1)
    
    a = pyproj.transform(lla, ecef, lon[0], lat[0], alt[0], radians=False)
    b = pyproj.transform(lla, ecef, lon[i2], lat[i2], alt[i2], radians=False)
    d = math.dist(b, a)
    
    dddd = d/d1
    dl_cor = np.append(dl_cor,dddd)
    # print(d/d1)
    # dl_cor.iloc[i1] = d/d1
    


tmp1 = tmp1 - tmp1[-1]*dl_cor
cpdmsp = np.round(max(tmp1/1000)-min(tmp1/1000))
cpcp1 = 'DMSP $\u03A6_{PC}$ = '+str(cpdmsp)+' kV'
ax1.text(1, 1.1, cpcp1, transform=ax1.transAxes, color='m')


vmin = 0.0;
vmax = 0.1;
norm = colors.Normalize(vmin=vmin, vmax=vmax)
ax1 = plt.subplot(3,2,4,projection='polar')
qcs1 = ax1.pcolormesh(yjh, xjh, zjh, norm=norm, cmap='Reds')
ax1.set_theta_zero_location("S")
ax1.set_ylim(0, 40)
ax1.set_title('(e) Joule Heating rate',fontweight="bold")
ax1.xaxis.grid(linestyle='--', linewidth=0.8)
ax1.yaxis.grid(linestyle='--', linewidth=0.8)
cbar1 = fig.colorbar(qcs1, ax=ax1,shrink=0.5,pad=0.1)
cbar1.set_label('$(W/m^2)$',fontweight="bold")
xtickpos = ax1.get_xticks()
xticks = ['00 MLT', '03','06','09','12','15','18','21']
ax1.set_xticks(xtickpos,xticks)
ytickpos = [5., 10., 15., 20., 25., 30., 35., 40.]
yticks = ['N', '80$^\circ$','','70$^\circ$','','60$^\circ$','','50$^\circ$']
ax1.set_yticks(ytickpos,yticks)

xfac,yfac,amp_pred,xsigp,ysigp,sigp,xsigh,ysigh,sigh,xpot,ypot,pot,xjh,yjh,zjh,xj,yj,z1,z2 = np.load('Figure2_data.npy')
x1,y1,z1,z2 = xj,yj,z1,z2
jhnorm = np.sqrt(z1**2 + z2**2)


vmin = 0;
vmax = 4;
norm = colors.Normalize(vmin=vmin, vmax=vmax)
cmap = 'seismic'
ax1 = plt.subplot(3,2,6,projection='polar')
qcs1 = ax1.pcolormesh(y1, x1, jhnorm, norm = norm, cmap='Reds')
x2, y2 = np.meshgrid(y1, x1)  # add these lines
U = z1
V = z2
num_arrows = len(x1) // 2

# Subsample the data by selecting every other arrow (skip one arrow each time)
subsampled_x1 = x1[::2][:num_arrows]
subsampled_y1 = y1[::2][:num_arrows]
subsampled_U = U[::2][:num_arrows]
subsampled_V = V[::2][:num_arrows]

# q = ax1.quiver(y1, x1, U, V, angles = 'xy', pivot='middle', scale=20, scale_units='width', width=0.005,  minlength=0.1, minshaft=0.1)
plt.quiver(subsampled_y1, subsampled_x1, subsampled_U, subsampled_V,
           angles='xy', pivot='middle', scale=20, scale_units='width',
           width=0.005, minlength=0.1, minshaft=0.1)

ax1.set_theta_zero_location("S")
# ax1.quiverkey(q, X=0.9, Y=1.05, U=0.5, label='Quiver key, length = 0.5', labelpos='E')
ax1.set_ylim(0, 40)
ax1.set_title('(f) Hall currents',fontweight="bold")
ax1.xaxis.grid(linestyle='--', linewidth=0.8)
ax1.yaxis.grid(linestyle='--', linewidth=0.8)
cbar1 = fig.colorbar(qcs1, ax=ax1,shrink=0.5,pad=0.1)
cbar1.set_label('(A/m)',fontweight="bold")
xtickpos = ax1.get_xticks()
xticks = ['00 MLT', '03','06','09','12','15','18','21']
ax1.set_xticks(xtickpos,xticks)
ytickpos = [5., 10., 15., 20., 25., 30., 35., 40.]
yticks = ['N', '80$^\circ$','','70$^\circ$','','60$^\circ$','','50$^\circ$']
ax1.set_yticks(ytickpos,yticks)
plt.tight_layout()
plt.savefig('Figure2.pdf',format='pdf')

