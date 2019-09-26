import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('AB_NYC_2019.csv')
data.head()

# PARSE THE DATAFRAME FOR ENTRIES PERTAINING TO DIFFERENT BOROUGHS
manhattan = data[data['neighbourhood_group'] == 'Manhattan']
manhattan.head()
brooklyn = data[data['neighbourhood_group'] == 'Brooklyn']
brooklyn.head()
queens = data[data['neighbourhood_group'] == 'Queens']
queens.head()


# DEMONSTRATE central limit theorem w/ histogram for avg price in manhattan 
trials = 2000
n = np.array([100,500,1000,2500,5000])
colors = np.array(['#37549E', '#ffce51', '#97CCFD', '#6281cf', '#97CCFD'])
f1 = plt.figure()
ax1 = f1.add_subplot(111)

for j in range(n.size):
    price_avg = np.zeros(trials)

    # Take 2000 samples of size n and find their avg price
    for i in range(trials):
        price_sample = (manhattan['price'].sample(n[j])).values
        price_avg[i] = np.mean(price_sample)
    
    # Plot each distribution for varying n    
    ax1.hist(price_avg, bins='auto', alpha = 0.7, color = colors[j],rwidth = 1.0, label = f'n = {n[j]}')
    plt.title('Average Airbnb Price in Manhattan')
    plt.xlabel('Average Nightly Price ($)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show

print('Note how the distribution becomes more normal as n increases')


# COMPARE avg price distributions in Manhattan, Brooklyn, & Queens
f2 = plt.figure()
ax2 = f2.add_subplot(111)

# Preallocate arrays for sampling data
manhattan_avg = np.zeros(trials)
brooklyn_avg = np.zeros(trials)
queens_avg = np.zeros(trials)

for i in range(trials):
    manhattan_avg[i] = np.mean((manhattan['price'].sample(n[3])).values)
    brooklyn_avg[i] = np.mean((brooklyn['price'].sample(n[3])).values)
    queens_avg[i] = np.mean((queens['price'].sample(n[3])).values)

# Plot the 3 histograms on the second figure
ax2.hist(manhattan_avg, bins='auto', color = '#6281cf', rwidth = 0.5, label = 'Manhattan')
ax2.hist(brooklyn_avg, bins='auto', color = '#ffce51', rwidth = 0.5, label = 'Brooklyn')
ax2.hist(queens_avg, bins='auto', color = '#97ccfd', rwidth = 0.5, label = 'Queens')

plt.title('Average Price by Borough')
plt.xlabel('Average Nightly Price ($)')
plt.ylabel('Frequency')
plt.legend()
plt.show 
