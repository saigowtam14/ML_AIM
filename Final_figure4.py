# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 10:04:12 2023

@author: SAI GOWTAM VALLURI
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from pandas import DataFrame
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")
from datetime import timedelta

year = 2017
month = 9
day1 = 7
day2 = 8
dmsp_id = 17

inpath = "inp_solar_imf"+str(year)+".txt"
start_tm = str(year)+'{:02d}'.format(month)+'{:02d}'.format(day1)+' 15:00:00'
end_tm = str(year)+'{:02d}'.format(month)+'{:02d}'.format(day2)+' 23:30:00'

f = open('SuperDARN'+str(year)+'{:02d}'.format(month)+'{:02d}'.format(day1)+'.pckl', 'rb')
sd = pickle.load(f)
f.close()
f = open('ocpcp'+str(year)+'{:02d}'.format(month)+'{:02d}'.format(day1)+'.pickle', 'rb')
cp = pickle.load(f)
f.close()
wei = pd.read_pickle('weimer'+str(year)+'{:02d}'.format(month)+'{:02d}'.format(day1))

f = open('dms_'+str(year)+"{:02d}".format(month)+"{:02d}".format(day1)+'_'+str(dmsp_id)+'.pckl', 'rb')
dmsp1 = pickle.load(f)
f.close()

f = open('dms_'+str(year)+"{:02d}".format(month)+"{:02d}".format(day1+1)+'_'+str(dmsp_id)+'.pckl', 'rb')
dmsp2 = pickle.load(f)
f.close()

dmsp = pd.concat([dmsp1,dmsp2])

start_tm = pd.to_datetime(start_tm)
end_tm = pd.to_datetime(end_tm)

df = pd.read_csv(inpath,skiprows=14,delim_whitespace=True);
df.columns = ['year','doy','hour','minute','Bx','By','Bz','Vx','Np','al','au','symh','asymh'];
df.replace(99999.9,np.NaN,inplace=True);
df.replace(9999.99,np.NaN,inplace=True);
df.replace(999.99,np.NaN,inplace=True);
df = df.interpolate();
df['datetime'] = pd.to_datetime(df['year'] * 1000 + df['doy'], format='%Y%j')
df['year'] = df.datetime.dt.year
df['month'] = df.datetime.dt.month
df['day'] = df.datetime.dt.day
df['month_sine'] = np.sin((2*np.pi*5)/12);
df['month_cosine'] = np.cos((2*np.pi*5)/12);
# f = (fluxdata['year']==df['year'][0])
df['F107'] = 113.2
dttime = DataFrame().assign(year = df['year'],month=df['month'],day = df['day'],hour = df['hour'],minute=df['minute'])
dttime['sec'] = 0;
df['datetime'] = pd.to_datetime(df[['month','day','year','hour','minute']]);
cols = ['datetime','Bz','By','Bx','Vx','Np','au','al','symh','asymh','F107','month_sine','month_cosine'];
df = df[cols];
tmpdf = df[cols];
df['datetime'] = pd.to_datetime(df['datetime'].astype(str))

tmpdf = tmpdf.set_index(['datetime'])
tmpdf = tmpdf.resample('2min').first()
tmpdf = tmpdf.reset_index()

df = df.set_index(['datetime'])
df = df.resample('2min').first()
df = df.reset_index()
intind = (df['datetime']>=start_tm) & (df['datetime']<=end_tm)
df = df.loc[intind]
tmpdf = tmpdf.loc[intind]
intind = (sd['datetime']>=start_tm) & (sd['datetime']<=end_tm)
sd = sd.loc[intind]

tm1 = pd.to_datetime('2017-09-07 22:02:00')
tm2 = pd.to_datetime('2017-09-07 23:42:00')
tm3 = pd.to_datetime('2017-09-08 15:04:00')
# tm4 = pd.to_datetime('2017-09-08 16:46:00')


fig,ax = plt.subplots(5,1,figsize=(20,15), dpi=300,gridspec_kw = {'wspace':0.025, 'hspace':0.025})
plt.rc('font', weight='bold',size=14)
ax[0].plot(tmpdf['datetime'], tmpdf['Bz'])
ax[0].plot(tmpdf['datetime'], tmpdf['By'])
ax[0].set_ylabel('IMF (nT)',weight='bold')
ax[0].grid(which='both', axis='both')
ax[0].set_xticklabels([])
ax[0].yaxis.set_tick_params(labelsize=14)
ax[0].set_ylim([-39,29])
ax[0].legend(['Bz','By'],loc='upper right',prop={'size': 14})
ax[0].text(0.03, 0.75, '(a)', transform = ax[0].transAxes)
ax[0].axvline(tm1, color="red", linestyle="dashed")
ax[0].axvline(tm2, color="red", linestyle="dashed")
ax[0].axvline(tm3, color="red", linestyle="dashed")
ax[0].xaxis.set_major_locator(mdates.DayLocator())
ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%m%d%y'))
ax[0].xaxis.set_minor_locator(mdates.HourLocator((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)))
# ax[0].xaxis.set_minor_formatter(mdates.DateFormatter('%H'))
ax[0].yaxis.set_ticks([-35,-30,-25,-20,-15,-10,-5,0,5,10,15,20,25])
# ax[0].xaxis.grid(True, which='minor')
# ax[0].yaxis.grid(True)
ax[0].set_xlim([start_tm,end_tm])
ax[0].fill_betweenx(ax[0].get_ylim(), tm1-timedelta(minutes=15), tm1, color='gray', alpha=0.3, label='Shaded Area')
ax[0].fill_betweenx(ax[0].get_ylim(), tm2-timedelta(minutes=15), tm2, color='gray', alpha=0.3, label='Shaded Area')
ax[0].fill_betweenx(ax[0].get_ylim(), tm3-timedelta(minutes=15), tm3, color='gray', alpha=0.3, label='Shaded Area')


ax[1].plot(tmpdf['datetime'], tmpdf['Vx'])
ax[1].set_ylabel('Vx (Km/s)',weight='bold')
ax[1].grid(which='both', axis='both')
ax[1].set_xticklabels([])
ax[1].yaxis.set_tick_params(labelsize=14)
ax[1].set_ylim([-1000,-100])
ax[1].text(0.03, 0.75, '(b)', transform = ax[1].transAxes)
ax[1].axvline(tm1, color="red", linestyle="dashed")
ax[1].axvline(tm2, color="red", linestyle="dashed")
ax[1].axvline(tm3, color="red", linestyle="dashed")
ax[1].xaxis.set_major_locator(mdates.DayLocator())
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
ax[1].xaxis.set_minor_locator(mdates.HourLocator((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)))
# ax[1].xaxis.set_minor_formatter(mdates.DateFormatter('%H'))
ax[1].yaxis.set_ticks([-800,-750,-700,-650,-600,-550,-500])
# ax[1].xaxis.grid(True, which='minor')
# ax[1].yaxis.grid(True)
ax[1].set_xlim([start_tm,end_tm])
ax[1].fill_betweenx(ax[1].get_ylim(), tm1-timedelta(minutes=15), tm1, color='gray', alpha=0.3, label='Shaded Area')
ax[1].fill_betweenx(ax[1].get_ylim(), tm2-timedelta(minutes=15), tm2, color='gray', alpha=0.3, label='Shaded Area')
ax[1].fill_betweenx(ax[1].get_ylim(), tm3-timedelta(minutes=15), tm3, color='gray', alpha=0.3, label='Shaded Area')

ax[2].plot(tmpdf['datetime'], tmpdf['Np'])
ax[2].set_ylabel('$Np/cm^3$',weight='bold',size = 14)
ax[2].set_ylim([0,13])
ax[2].grid(which='both', axis='both')
ax[2].set_xticklabels([])
ax[2].yaxis.set_tick_params(labelsize=14)
ax[2].text(0.03, 0.8, '(c)', transform = ax[2].transAxes)
ax[2].axvline(tm1, color="red", linestyle="dashed")
ax[2].axvline(tm2, color="red", linestyle="dashed")
ax[2].axvline(tm3, color="red", linestyle="dashed")
ax[2].xaxis.set_major_locator(mdates.DayLocator())
ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%m%d%y'))
ax[2].xaxis.set_minor_locator(mdates.HourLocator((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)))
# ax[2].xaxis.set_minor_formatter(mdates.DateFormatter('%H'))
# ax[2].xaxis.grid(True, which='minor')
# ax[2].yaxis.grid(True)
ax[2].set_xlim([start_tm,end_tm])
ax[2].fill_betweenx(ax[2].get_ylim(), tm1-timedelta(minutes=15), tm1, color='gray', alpha=0.3, label='Shaded Area')
ax[2].fill_betweenx(ax[2].get_ylim(), tm2-timedelta(minutes=15), tm2, color='gray', alpha=0.3, label='Shaded Area')
ax[2].fill_betweenx(ax[2].get_ylim(), tm3-timedelta(minutes=15), tm3, color='gray', alpha=0.3, label='Shaded Area')


ax[3].plot(tmpdf['datetime'], tmpdf['symh'])
ax[3].set_ylabel('nT',weight='bold')
ax[3].set_ylim([-200,50])
ax[3].grid(which='both', axis='both')
ax[3].legend(['Sym-H'],loc='upper right',prop={'size': 14})
ax[3].set_xticklabels([])
# ax[3].set_ylim([-40,20])
ax[3].yaxis.set_tick_params(labelsize=14)
ax[3].text(0.03, 0.8, '(d)', transform = ax[3].transAxes)
ax[3].axvline(tm1, color="red", linestyle="dashed")
ax[3].axvline(tm2, color="red", linestyle="dashed")
ax[3].axvline(tm3, color="red", linestyle="dashed")
ax[3].xaxis.set_major_locator(mdates.DayLocator())
ax[3].xaxis.set_major_formatter(mdates.DateFormatter('%m%d%y'))
ax[3].xaxis.set_minor_locator(mdates.HourLocator((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)))
# ax[3].xaxis.set_minor_formatter(mdates.DateFormatter('%H'))
ax[3].yaxis.set_ticks([-150,-125,-100,-75,-50,-25,0])
# ax[3].xaxis.grid(True, which='minor')
# ax[3].yaxis.grid(True)
ax[3].set_xlim([start_tm,end_tm])
ax[3].fill_betweenx(ax[3].get_ylim(), tm1-timedelta(minutes=15), tm1, color='gray', alpha=0.3, label='Shaded Area')
ax[3].fill_betweenx(ax[3].get_ylim(), tm2-timedelta(minutes=15), tm2, color='gray', alpha=0.3, label='Shaded Area')
ax[3].fill_betweenx(ax[3].get_ylim(), tm3-timedelta(minutes=15), tm3, color='gray', alpha=0.3, label='Shaded Area')


errors = [timedelta(minutes=15), timedelta(minutes=15),timedelta(minutes=15), timedelta(minutes=15),
          timedelta(minutes=15), timedelta(minutes=15),timedelta(minutes=15), timedelta(minutes=15),
          timedelta(minutes=15), timedelta(minutes=15),timedelta(minutes=15), timedelta(minutes=15),
          timedelta(minutes=15), timedelta(minutes=15),timedelta(minutes=15), timedelta(minutes=15),
          timedelta(minutes=15), timedelta(minutes=15),timedelta(minutes=15), timedelta(minutes=15),
          timedelta(minutes=15), timedelta(minutes=15),timedelta(minutes=15), timedelta(minutes=15),
          timedelta(minutes=15), timedelta(minutes=15),timedelta(minutes=15), timedelta(minutes=15)]
ax[4].plot(cp['timestamps'],cp['ML-AIM'],zorder=1)
ax[4].plot(sd['datetime'],sd['SuperDARN'],zorder=2)
ax[4].plot(wei['datetime'],wei['Weimer2005'],zorder=3)
ax[4].scatter(dmsp['datetime'],dmsp['DMSP'],color = 'm',zorder=4)
ax[4].errorbar(dmsp['datetime'],dmsp['DMSP'],xerr=errors, fmt='none', ecolor='m', capsize=3)
ax[4].set_ylabel('$\u03A6_{PC}$ (kV)',weight='bold')
ax[4].grid(which='both', axis='both')
ax[4].legend(['ML-AIM','SuperDARN','Weimer2005','DMSP'],loc="upper center", ncol=4, prop={'size': 14})
ax[4].yaxis.set_tick_params(labelsize=14)
ax[4].set_xlabel('Date and Universal Time (Hours)',weight='bold')
ax[4].set_ylim([10,300])
ax[4].text(0.03, 0.8, '(e)', transform = ax[4].transAxes)
# ax[4].text()
ax[4].axvline(tm1, color="red", linestyle="dashed")
ax[4].axvline(tm2, color="red", linestyle="dashed")
ax[4].axvline(tm3, color="red", linestyle="dashed")
ax[4].xaxis.set_major_locator(mdates.DayLocator())
ax[4].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax[4].xaxis.set_minor_locator(mdates.HourLocator((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)))
ax[4].xaxis.set_minor_formatter(mdates.DateFormatter('%H'))
ax[4].yaxis.set_ticks([50,100,150,200,250])
# ax[4].xaxis.grid(True, which='minor')
# ax[4].yaxis.grid(True)
ax[4].set_xlim([start_tm,end_tm])

plt.savefig('Figure4.pdf',format='pdf')