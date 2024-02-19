# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 09:33:10 2023

@author: SAI GOWTAM VALLURI
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from pandas import DataFrame
from datetime import timedelta
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

year = 2013
month = 5
day1 = 14
dmsp_id = 17

inpath = "inp_solar_imf"+str(year)+".txt"
start_tm = str(year)+'{:02d}'.format(month)+'{:02d}'.format(day1)+' 05:00:00'
end_tm = str(year)+'{:02d}'.format(month)+'{:02d}'.format(day1)+' 23:30:00'

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

# f = open('dms_'+str(year)+"{:02d}".format(month)+"{:02d}".format(day1+1)+'_'+str(dmsp_id)+'.pckl', 'rb')
# dmsp2 = pickle.load(f)
# f.close()

dmsp = dmsp1

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

fig,ax = plt.subplots(4,1,figsize=(20,15), dpi=600, gridspec_kw = {'wspace':0.025, 'hspace':0.025})
ax[0].plot(tmpdf['datetime'],tmpdf['Bz'])
ax[0].plot(tmpdf['datetime'],tmpdf['By'])
ax[0].set_ylabel('IMF (nT)',weight='bold',size=14)
ax[0].grid(which='both', axis='both')
ax[0].set_xticklabels([])
ax[0].yaxis.set_tick_params(labelsize=14)
ax[0].set_ylim([-9,9])
ax[0].legend(['Bz','By'],loc='upper right',prop={'size': 14,'weight':'bold'})
ax[0].text(0.01, 0.9, '(a)', transform = ax[0].transAxes,size = 14, weight = 'bold')
tm1 = pd.to_datetime('2013-05-14 12:44')
tm2 = pd.to_datetime('2013-05-14 11:44')
ax[0].axvline(tm1, color="red", linestyle="dashed")
ax[0].fill_betweenx(ax[0].get_ylim(), tm2, tm1, color='gray', alpha=0.3, label='Shaded Area')
# ax[0].xaxis.set_major_locator(mdates.DayLocator())
# ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%m%d%y'))
ax[0].xaxis.set_minor_locator(mdates.HourLocator((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)))
ax[0].xaxis.set_minor_formatter(mdates.DateFormatter('%H'))
ax[0].set_xlim([start_tm+timedelta(hours=1),end_tm])


# tmpdf.plot(x = 'datetime', y = ['Vx'],ax=ax[1])
ax[1].plot(tmpdf['datetime'],tmpdf['Vx'])
ax[1].set_ylabel('Vx (Km/s)',weight='bold',size=14)
ax[1].grid(which='both', axis='both')
ax[1].legend(['Vx'],loc='upper right',prop={'size': 14,'weight':'bold'})
ax[1].set_xticklabels([])
ax[1].yaxis.set_tick_params(labelsize=14)
ax[1].set_ylim([-420,-310])
ax[1].text(0.01, 0.9, '(b)', transform = ax[1].transAxes,size = 14, weight = 'bold')
# ax[1].xaxis.set_major_locator(mdates.DayLocator())
# ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%m%d%y'))
ax[1].axvline(tm1, color="red", linestyle="dashed")
ax[1].xaxis.set_minor_locator(mdates.HourLocator((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)))
ax[1].xaxis.set_minor_formatter(mdates.DateFormatter('%H'))
ax[1].set_xlim([start_tm+timedelta(hours=1),end_tm])
ax[1].fill_betweenx(ax[1].get_ylim(), tm2, tm1, color='gray', alpha=0.3, label='Shaded Area')

# tmpdf.plot(x = 'datetime', y = ['Np'],ax=ax[2])
ax[2].plot(tmpdf['datetime'],tmpdf['Np'])
ax[2].set_ylabel('$Np/cm^3$',weight='bold',size =14)
ax[2].set_ylim([4,13])
ax[2].grid(which='both', axis='both')
ax[2].legend(['Np'],loc='upper right',prop={'size': 14,'weight':'bold'})
ax[2].set_xticklabels([])
ax[2].yaxis.set_tick_params(labelsize=14)
ax[2].text(0.01, 0.9, '(c)', transform = ax[2].transAxes,size = 14, weight = 'bold')
# ax[2].xaxis.set_major_locator(mdates.DayLocator())
# ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%m%d%y'))
ax[2].axvline(tm1, color="red", linestyle="dashed")
ax[2].xaxis.set_minor_locator(mdates.HourLocator((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)))
ax[2].xaxis.set_minor_formatter(mdates.DateFormatter('%H'))
ax[2].set_xlim([start_tm+timedelta(hours=1),end_tm])
ax[2].fill_betweenx(ax[2].get_ylim(), tm2, tm1, color='gray', alpha=0.3, label='Shaded Area')

errors = [timedelta(minutes=15), timedelta(minutes=15),timedelta(minutes=15), timedelta(minutes=15),
          timedelta(minutes=15), timedelta(minutes=15),timedelta(minutes=15), timedelta(minutes=15),
          timedelta(minutes=15), timedelta(minutes=15),timedelta(minutes=15), timedelta(minutes=15),
          timedelta(minutes=15)]


ax[3].plot(cp['timestamps'],cp['ML-AIM'])
ax[3].plot(sd['datetime'],sd['SuperDARN'])
ax[3].plot(wei['datetime'],wei['Weimer2005'])
ax[3].scatter(dmsp['datetime'],dmsp['DMSP'],color='m')
ax[3].errorbar(dmsp['datetime'],dmsp['DMSP'],xerr=errors, fmt='none', ecolor='m', capsize=3)
ax[3].set_ylabel('$\u03A6_{PC}$ (kV)',weight='bold',size=14)
ax[3].grid(which='both', axis='both')
ax[3].legend(['ML-AIM','SuperDARN','Weimer2005','DMSP'],loc="upper center", ncol=4, prop={'size': 14,'weight':'bold'})
ax[3].set_xlabel('Universal Time (Hours)',weight='bold',size=14)
ax[3].set_ylim([0,150])
ax[3].text(0.01, 0.9, '(d)', transform = ax[3].transAxes,weight='bold',size=14)
ax[3].xaxis.set_major_locator(mdates.DayLocator())
ax[3].yaxis.set_tick_params(labelsize=14)
hours_formatter = mdates.DateFormatter('%H:%M')
ax[3].xaxis.set_major_locator(mdates.HourLocator((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)))
ax[3].xaxis.set_major_formatter(hours_formatter)
ax[3].set_xlim([start_tm+timedelta(hours=1),end_tm])

plt.savefig('Figure3.pdf',format='pdf')