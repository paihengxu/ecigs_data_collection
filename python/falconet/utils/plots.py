import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import json
from pandas.io.json import json_normalize
import collections
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.basemap import Basemap as Basemap

us = ["alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
"delaware", "district of columbia", "florida", "georgia", "hawaii", "idaho", "illinois",
"indiana", "iowa", "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts",
"michigan", "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada", 
"new Hampshire", "new Jersey", "new Mexico", "new York", "north Carolina", "north Dakota",
"ohio", "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina", 
"south dakota", "tennessee", "texas", "utah", "vermont", "virginia", "washington", 
"west Virginia", "wisconsin", "wyoming"]

BLUE_M = "#85C1E9"
GREEN_M = "#45b39d"

def __plot_map(ax, data, vmin=0, vmax=100,cmap=plt.cm.GnBu_r ):
    # Lambert Conformal map of lower 48 states.
    m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
    #read shape information
    shp_info = m.readshapefile('../resources/st99_d00','states',drawbounds=True)
    colors={}
    statenames=[]    
    # setup the colorbar
    cmap = cmap
    normalize = mcolors.Normalize(vmin=vmin, vmax=vmax)    
    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
    scalarmappaple.set_array([])
    
    for shapedict in m.states_info:
        statename = shapedict['NAME']
        statenames.append(statename)
        if statename not in ['Alaska','Puerto Rico']:
            try:    
                pop = data[statename.lower()]
                #print statename, " > ", pop
                colors[statename] = cmap(normalize(pop))
            except (IndexError, KeyError):
                print("skipping " + statename)
                continue        
            
    # cycle through state names, color each one.
    for nshape,seg in enumerate(m.states):
        if statenames[nshape] not in ['Alaska','Puerto Rico']:
            try:
                color = rgb2hex(colors[statenames[nshape]]) 
                poly = Polygon(seg,facecolor=color,edgecolor=color)
                ax.add_patch(poly)
            except (IndexError, KeyError):
    #            print "skipping ", statenames[nshape]
                continue
    return ax, scalarmappaple

def plot_by_state(data, ax):
    data["location-state"] = [x.lower() for x in data["location-state"]]
    states = data[data["location-state"].isin(us)]
    states = states.groupby(['location-state']).size()
    states /= sum(states)
    ax, cbar = __plot_map(ax, states.to_dict(), vmax=0.2)
    return ax, cbar

def plot_aggregation(data, aggregation, ax, colors):
    assert isinstance(aggregation, list)    
    agg=data.groupby(aggregation).size()
    ax = agg.plot(kind="bar", ax=ax,color=colors)
    return ax
    
def plot_monthly(data, ax, color=GREEN_M):
    #add a month field for the aggregations
    months = pd.to_datetime(data['timestamp']).dt.to_period('M')
    data["month"] = months
    monthly = data.groupby(['month']).size()
    ax = monthly.plot(kind="bar", ax=ax, color=color)
    return ax

def plot_overtime(data, ax, color=GREEN_M):
    overtime = data.groupby(['timestamp']).size()
    ax = overtime.plot(kind="bar", ax=ax, color=color)
    return ax
