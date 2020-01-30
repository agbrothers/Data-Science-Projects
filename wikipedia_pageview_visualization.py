import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['font.weight'] = 'bold'
import matplotlib.animation as animation


# Visualizing wikipedia pageview data from a previous project with an animated bar chart

""" DATA """

data = pd.read_csv("alldata.csv")

number_of_frames = data.shape[0]*2
frame_rate = 50  # how fast do you want info to fly at you? 25
fig,ax = plt.subplots(figsize=(15,8))
x_range = int(4e+6)

Names = [data.iloc[i][3:13] for i in data.index]
Values = [data.iloc[i][13:] for i in data.index]


""" BAR CHART DRAWING FUNCTION """

def draw(frame, data):
    ax.clear()
    ax.barh(np.arange(len(Values[frame])), Values[frame], ec='k', color='#82baef',
            align='center', alpha = 0.6, linewidth = 2)
    
    """ AXIS LABELS """
    clr = '#363636'
    ax.invert_yaxis()
    ax.set_yticks(np.arange(0))
    ax.set_xlabel('\n\nView Count per Day', color=clr, size=12)
    ax.tick_params(axis='x', colors=clr, labelsize=10)              
    ax.set_title('HIGHEST VIEWED WIKIPEDIA PAGES\n', color=clr, size=12, weight='bold')
    ax.text(0.009,1.05, f'{data.Month[frame]}.{data.Day[frame]}.{data.Year[frame]}', transform=ax.transAxes, 
            ha='left', va='bottom', color=clr, bbox=dict(fc='w', alpha=0.6, ec='k', linewidth='2', pad=6))
    
    """ BAR LABELS """  
    for i, (value, name) in enumerate(zip(Values[frame], Names[frame])):
        ax.text(value, i, '  ' + name, size=10, weight=600, ha='left', va='bottom')
        ax.text(value, i+0.3, '   ' + str(value), size=8, color='#444444', ha='left', va='bottom')
    
    """ AXIS FORMAT """
    plt.xticks(rotation=30)  # angle of x-axis labels
    plt.box(False)
    ax.set_xlim(0, x_range)
    ax.margins(0, 0.01)
    ax.grid(which='major', axis='x', linestyle='-', color='#DCDCDC')
    ax.set_axisbelow(True)
    plt.gcf().subplots_adjust(left=0.02, bottom=0.2)

# Produce the Animatoin
anim = animation.FuncAnimation(fig, draw, number_of_frames, fargs=(data, ), interval=(1/frame_rate)*1000, blit=False)
plt.show()  

# Format & save the animation file
FFwriter=animation.FFMpegWriter(fps=10, extra_args=['-vcodec', 'libx264'])
anim.save('plswork.mp4', writer=FFwriter, dpi=100)
