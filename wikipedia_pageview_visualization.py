import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
import matplotlib.animation as animation

# Visualizing wikipedia top pageview data from a previous project with an animated bar chart

""" DATA """

data = pd.read_csv("alldata.csv")

number_of_frames = data.shape[0]
frame_rate = 1
fig,ax = plt.subplots(figsize=(15,8))


""" BAR CHART DRAWING FUNCTION """

def draw(frames, data):
    ax.clear()
    ax.barh(np.arange(len(data.iloc[frames][13:])), data.iloc[frames][13:], ec='k', color='#82baef',
            align='center', alpha = 0.6, linewidth = 2)
    
    """ LABELS """
    clr = '#777777'
    ax.invert_yaxis()
    ax.set_yticks(np.arange(len(data.iloc[frames][3:13])))
    ax.set_yticklabels(data.iloc[frames][3:13], color=clr, size=12)
    ax.set_xlabel('\nView Count per Day', color=clr, size=12)
    ax.tick_params(axis='x', colors=clr, labelsize=12)              
    ax.set_title('HIGHEST VIEWED WIKIPEDIA PAGES\n', color=clr, size=12)
    ax.text(0.005,1.05, f'DATE: {data.Month[frames]}.{data.Day[frames]}.{data.Year[frames]}', transform=ax.transAxes, 
            ha='left', color='w', bbox=dict(facecolor='#82baef', alpha=0.8, edgecolor='w'))
    
    """ AXIS FORMAT """
    plt.xticks(rotation=50)
    plt.box(False)
    ax.set_xlim(0,2000000)
    ax.margins(0, 0.01)
    ax.grid(which='major', axis='x', linestyle='-')
    ax.set_axisbelow(True)
    plt.gcf().subplots_adjust(left=0.3, bottom=0.2)

anim = animation.FuncAnimation(fig, draw, number_of_frames, fargs=(data, ), interval=(1/frame_rate)*1000, blit=False)
plt.show()  

#Set up formatting for the movie files
Writer = animation.writers['pillow']
writer = Writer(fps=frame_rate, metadata=dict(artist='Me'), bitrate=1800)
anim.save('mymovie.mp4',writer=writer)
