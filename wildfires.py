import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import randn


""" PULL DATA FROM SQLite DATABASE """
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
data = data[data['STATE'] == 'CA']


""" DATA VISUALIZATION """
"""   HEATMAP BY DOY   """
# visualize when fires occur throughout the year & how it changes YoY
fig1 = plt.figure()
years = list(range(min(data.FIRE_YEAR), max(data.FIRE_YEAR),1))
heatmap1 = pd.DataFrame()

for year in years:
    df1 = data[data['FIRE_YEAR'] == year]
    row = pd.DataFrame([np.bincount(df1.DISCOVERY_DOY)], index=[year])
    heatmap1 = heatmap1.append(row)

heatmap1 = heatmap1.drop(columns=[0,366])
heatmap1 = heatmap1.replace(np.nan, 0.0)
sns.heatmap(heatmap1)
plt.gca().invert_yaxis()
plt.xlabel("\nDay of Year")
plt.ylabel("Year\n")
plt.title("Distribution of California Fires by Day of Year\n")
plt.tight_layout()
plt.show()


""" HEATMAP BY FIRE SIZE """
# Visualize when fires of a given size occur throughout the year
fig2 = plt.figure()
sizes = list(set(data.FIRE_SIZE_CLASS))
sizes = ['A','B','C','D','E','F','G']   # Classes by increasing size (A smallest, G largest)
heatmap2 = pd.DataFrame()

for size in sizes:
    print(size)
    df2 = data[data['FIRE_SIZE_CLASS'] == size]
    row = pd.DataFrame([np.bincount(df2.DISCOVERY_DOY)], index=[size])
    heatmap2 = heatmap2.append(row)

heatmap2 = heatmap2.drop(columns=[0,366])
heatmap2 = heatmap2.replace(np.nan, 0.0)
sns.heatmap(heatmap2)
# plt.gca().invert_yaxis()
plt.xlabel("\nDAY OF YEAR")
plt.ylabel("SIZE CLASS\n")
plt.title("DISTRIBUTION OF FIRE SIZE BY DOY\n")
plt.tight_layout()
plt.show()


""" HEATMAP BY LARGE FIRE SIZE """
# Larger fires are far less frequent, this heatmap highlights the distribution of those fires
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
heatmap3 = heatmap3.replace(np.nan, 0.0)
sns.heatmap(heatmap3)
# plt.gca().invert_yaxis()
plt.xlabel("\nDAY OF YEAR")
plt.ylabel("SIZE CLASS\n")
plt.title("DISTRIBUTION OF LARGE FIRES BY DOY\n")
plt.tight_layout()
plt.show()
