import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
# from mpl_toolkits.basemap import basemap
from numpy.random import randn

"""
# CONNECT to the SQLite Database
conn = sqlite3.connect("FPA_FOD_20170508.sqlite")
c = conn.cursor()

names1 = "FIRE_NAME, FIRE_YEAR, DISCOVERY_DATE, DISCOVERY_DOY, DISCOVERY_TIME"
names2 = "STAT_CAUSE_CODE, STAT_CAUSE_DESCR, CONT_DATE, CONT_TIME"
names3 = "FIRE_SIZE, FIRE_SIZE_CLASS, STATE, COUNTY, FIPS_NAME"

# Populate a dataframe with table info from our database
c.execute(f'SELECT {names1},{names2},{names3} from FIRES')
rawd = c.fetchall()''
data = pd.DataFrame(rawd, columns=['FIRE_NAME','FIRE_YEAR','DISCOVERY_DATE', 
                                   'DISCOVERY_DOY','DISCOVERY_TIME','STAT_CAUSE_CODE',
                                   'STAT_CAUSE_DESCR','CONT_DATE','CONT_TIME',
                                   'FIRE_SIZE','FIRE_SIZE_CLASS','STATE','COUNTY','FIPS_NAME'])
data.to_csv('wildfires.csv')
"""

# After already loading and saving SQLITE data
data = pd.read_csv('wildfires.csv')

# Data Vis
"""
# Nine random samples of n=10000 fires, looking to see when the largest fires occur
fig = plt.figure(figsize=(11,8))
ax1 = fig.add_subplot(3,3,1)
ax2 = fig.add_subplot(3,3,2)
ax3 = fig.add_subplot(3,3,3)
ax4 = fig.add_subplot(3,3,4)
ax5 = fig.add_subplot(3,3,5)
ax6 = fig.add_subplot(3,3,6)
ax7 = fig.add_subplot(3,3,7)
ax8 = fig.add_subplot(3,3,8)
ax9 = fig.add_subplot(3,3,9)


plots = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]
for ax in plots:
    df = data.sample(n=10000) # 10000 fire random sample
    
    # Remove outlier fires of size > 200
    df.drop(df[df.FIRE_SIZE > 200].index, inplace=True) 
    df.drop(df[df.FIRE_SIZE < 5].index, inplace=True) 
    ax.scatter(x=df.DISCOVERY_DOY, y=df.FIRE_SIZE, c='#FFAC2F', marker='.', alpha=0.3)
    
    plt.xlabel("Day of Year Discovered")
    plt.ylabel("Fire Size")
    
plt.tight_layout()
plt.show()
"""


""" HEATMAP BY DOY """
fig1 = plt.figure()
years = list(range(min(data.FIRE_YEAR), max(data.FIRE_YEAR),1))
heatmap1 = pd.DataFrame()

for year in years:
    df1 = data[data['FIRE_YEAR'] == year]
    row = pd.DataFrame([np.bincount(df1.DISCOVERY_DOY)], index=[year])
    heatmap1 = heatmap1.append(row)

heatmap1 = heatmap1.drop(columns=[0,366])
sns.heatmap(heatmap1)
plt.gca().invert_yaxis()
plt.xlabel("\nDay of Year")
plt.ylabel("Year\n")
plt.title("Distribution of California Fires by Day of Year\n")
plt.tight_layout()
plt.show()


""" HEATMAP BY FIRE SIZE """
fig2 = plt.figure()
sizes = list(set(data.FIRE_SIZE_CLASS))
sizes = ['A','B','C','D','E','F','G']
heatmap2 = pd.DataFrame()

for size in sizes:
    print(size)
    df2 = data[data['FIRE_SIZE_CLASS'] == size]
    row = pd.DataFrame([np.bincount(df2.DISCOVERY_DOY)], index=[size])
    heatmap2 = heatmap2.append(row)

heatmap2 = heatmap2.drop(columns=[0,366])
sns.heatmap(heatmap2)
# plt.gca().invert_yaxis()
plt.xlabel("\nDAY OF YEAR")
plt.ylabel("SIZE CLASS\n")
plt.title("DISTRIBUTION OF FIRE SIZE BY DOY\n")
plt.tight_layout()
plt.show()


""" HEATMAP BY LARGE FIRE SIZE """
fig3 = plt.figure()
sizes = list(set(data.FIRE_SIZE_CLASS))
sizes = ['D','E','F','G']
heatmap3 = pd.DataFrame()

for size in sizes:
    print(size)
    df2 = data[data['FIRE_SIZE_CLASS'] == size]
    row = pd.DataFrame([np.bincount(df2.DISCOVERY_DOY)], index=[size])
    heatmap3 = heatmap3.append(row)

heatmap3 = heatmap3.drop(columns=[0,366])
sns.heatmap(heatmap3)
# plt.gca().invert_yaxis()
plt.xlabel("\nDAY OF YEAR")
plt.ylabel("SIZE CLASS\n")
plt.title("DISTRIBUTION OF LARGE FIRES BY DOY\n")
plt.tight_layout()
plt.show()

             
# Bubble Plot



# Bubble Map                          
             
             