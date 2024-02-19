# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:00:22 2024

@author: SAI GOWTAM VALLURI
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

test = pd.read_csv('2017DMSP.csv')


ML_rmse = np.sqrt(np.mean((test['DMSP-CPCP'] - test['ML_AIM-CPCP'])**2))
wei_rmse = np.sqrt(np.mean((test['DMSP-CPCP'] - test['Weimer2005'])**2))
sd_rmse = np.sqrt(np.mean((test['DMSP-CPCP'] - test['SuperDARN'])**2))

ML_mae = mean_absolute_error(test['DMSP-CPCP'], test['ML_AIM-CPCP'])
wei_mae = mean_absolute_error(test['DMSP-CPCP'], test['Weimer2005'])
sd_mae = mean_absolute_error(test['DMSP-CPCP'], test['SuperDARN'])


ml_slope, ml_intercept = np.polyfit(test['DMSP-CPCP'], test['ML_AIM-CPCP'], 1)
wei_slope, wei_intercept = np.polyfit(test['DMSP-CPCP'], test['Weimer2005'], 1)
sd_slope, sd_intercept = np.polyfit(test['DMSP-CPCP'], test['SuperDARN'], 1)

ML_corr = np.corrcoef(test['DMSP-CPCP'], test['ML_AIM-CPCP'])[0, 1] if len(test) > 1 else np.nan
wei_corr = np.corrcoef(test['DMSP-CPCP'], test['Weimer2005'])[0, 1] if len(test) > 1 else np.nan
sd_corr = np.corrcoef(test['DMSP-CPCP'], test['SuperDARN'])[0, 1] if len(test) > 1 else np.nan



column_names = ['year', 'doy', 'hour', 'kp', 'ae', 'al', 'au']
slrind = pd.read_csv('kp2017.txt', delimiter='\s+', names = column_names)

merged_df = pd.merge(test, slrind, on=['year', 'doy', 'hour'])

merged_df['kp'] = merged_df['kp']/10


grouped_data = merged_df.groupby('kp')
counts = merged_df['kp'].value_counts()
result_df = counts.reset_index()
result_df.columns = ['kp', 'Count']
result_df = result_df.sort_values(by='kp')

mean_ML_AIM = grouped_data['ML_AIM-CPCP'].mean()
std_ML_AIM = grouped_data['ML_AIM-CPCP'].std()

mean_Weimer2005 = grouped_data['Weimer2005'].mean()
std_Weimer2005 = grouped_data['Weimer2005'].std()

mean_SuperDARN = grouped_data['SuperDARN'].mean()
std_SuperDARN = grouped_data['SuperDARN'].std()

mean_DMSP = grouped_data['DMSP-CPCP'].mean()
std_DMSP = grouped_data['DMSP-CPCP'].std()

params = {
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'lines.linewidth': 2.5,
    'errorbar.capsize': 12,
    'font.weight': 'bold'
}
plt.rcParams.update(params)
fig, axs = plt.subplots(figsize=(12, 10), dpi=600)

x_values = np.linspace(20, 250, 100)  # Adjust the range as needed
axs.plot(x_values, x_values, color='black', linestyle='--', label='y=x')
axs.scatter(test['DMSP-CPCP'], test['Weimer2005'], alpha=0.7, color='green', label='Weimer2005')
axs.scatter(test['DMSP-CPCP'], test['SuperDARN'], alpha=0.7, color='orange', label='SuperDARN')
axs.scatter(test['DMSP-CPCP'], test['ML_AIM-CPCP'], alpha=0.7, color='#1f77b4', label='ML_AIM')

axs.plot(test['DMSP-CPCP'], wei_slope * test['DMSP-CPCP'] + wei_intercept, color='green', linestyle='--', label='Weimer2005 Linear Fit')
axs.plot(test['DMSP-CPCP'], sd_slope * test['DMSP-CPCP'] + sd_intercept, color='orange', linestyle='--', label='SuperDARN Linear Fit')
axs.plot(test['DMSP-CPCP'], ml_slope * test['DMSP-CPCP'] + ml_intercept, color='#1f77b4', linestyle='--', label='ML_AIM Linear Fit')

axs.set_xlabel('Observed $\u03A6_{PC}$', fontweight='bold')
axs.set_ylabel('Predicted $\u03A6_{PC}$', fontweight='bold')

axs.text(22, 230, 'RMSE', size=20, weight='bold')
axs.text(22, 220, '24keV', size=20, weight='bold', color = '#1f77b4')
axs.text(22, 210, '28keV', size=20, weight='bold', color = 'green')
axs.text(22, 200, '27keV', size=20, weight='bold', color = 'orange')

axs.text(65, 230, 'MAE', size=20, weight='bold')
axs.text(65, 220, '18keV', size=20, weight='bold', color = '#1f77b4')
axs.text(65, 210, '22keV', size=20, weight='bold', color = 'green')
axs.text(65, 200, '19keV', size=20, weight='bold', color = 'orange')

axs.text(106, 230, 'Corr. Coeff.', size=20, weight='bold')
axs.text(106, 220, '0.45', size=20, weight='bold', color = '#1f77b4')
axs.text(106, 210, '0.39', size=20, weight='bold', color = 'green')
axs.text(106, 200, '0.35', size=20, weight='bold', color = 'orange')

axs.set_xlim([20, 250])
axs.set_ylim([20, 250])
axs.legend(loc='upper right')
plt.tight_layout()

plt.savefig('Figure6.pdf',format='pdf')
plt.show()



params = {
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'lines.linewidth': 2.5,
    'errorbar.capsize': 12,
    'font.weight': 'bold'
}
plt.rcParams.update(params)
fig, axs = plt.subplots(figsize=(12, 10), dpi=600)
bar_width = 0.2
axs.bar(result_df['kp'], result_df['Count'], color='gray', alpha=0.5, label='Count', width=bar_width, align='center')

# Creating twin axes for the second y-axis
axs1_y2 = axs.twinx()

# Plotting on the twin axes
axs1_y2.errorbar(mean_ML_AIM.index, mean_ML_AIM, yerr=std_ML_AIM, fmt='--o', color='#1f77b4', label='ML_AIM', capsize=10, markersize=10)
axs1_y2.errorbar(mean_Weimer2005.index, mean_Weimer2005, yerr=std_Weimer2005, fmt='--o', color='green', label='Weimer2005', capsize=10, markersize=10)
axs1_y2.errorbar(mean_SuperDARN.index, mean_SuperDARN, yerr=std_SuperDARN, fmt='--o', color='orange', label='SuperDARN', capsize=10, markersize=10)
axs1_y2.errorbar(mean_DMSP.index, mean_DMSP, yerr=std_DMSP, fmt='--o', color='black', label='DMSP', capsize=10, markersize=10)

# Setting labels and limits for the second y-axis
axs1_y2.set_ylabel('$\u03A6_{PC}$', fontweight='bold', size=20)
axs1_y2.tick_params(axis='y')

# Setting labels and limits for the x-axis
axs.set_xlabel('Kp-index', fontweight='bold', size=20)
axs.set_ylabel('Count', fontweight='bold', size=20)
axs.set_xticks([0,1,2,3,4,5,6,7,8,9])
axs.set_xlim([-0.5, 9])

# Adding legend for both y-axes in the second subplot
lines, labels = axs.get_legend_handles_labels()
lines2, labels2 = axs1_y2.get_legend_handles_labels()
axs.legend(lines + lines2, labels + labels2, loc='upper right')

plt.tight_layout()

plt.savefig('Figure7.pdf',format='pdf')
plt.show()