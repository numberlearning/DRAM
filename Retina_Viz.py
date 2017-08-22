print("Setting everything up!")
import warnings
warnings.filterwarnings('ignore')

from bokeh.io import push_notebook, show, output_notebook
from bokeh.layouts import row, column
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, CustomJS, FixedTicker
import bokeh.palettes as pal
from bokeh.layouts import layout, Spacer, gridplot
output_notebook()

#import ipywidgets as widgets
#from ipywidgets import *
from IPython.display import display, clear_output

import numpy as np

from create_data_test_newfilterbank import num_img, origin_img, filter_img, gx, gy, read_n

clear_output()
#b = Button(description="Loading...", icon="arrow", width=400)

figures = list()
iqs = list()

def make_spacer():
    p = figure(x_range=(-0.5, 10), y_range=(0, 1), width=200, height=350, tools="", border_fill_alpha=0, outline_line_alpha=0)
    p.toolbar.logo = None
    p.toolbar_location = None
    p.border_fill_alpha = 0
    p.outline_line_alpha = 0
    
    p.axis.visible = False
    p.grid.visible = False
    
    return p

def make_figure(color, hover=False):
    """
    color: attention window color
    hover: if the image is hovered
    """
    if hover:
        w = read_n
        img = filter_img
        title = "Hover Image"
    else:
        w = 100
        img = origin_img
        title = "Unhover Image"
        
    p = figure(x_range=(0, w), y_range=(w, 0), width=350, height=350, tools="", title=title)
    
    p.toolbar.logo = None
    p.toolbar_location = None
    p.axis.visible = False

    i_source = ColumnDataSource(data=dict(image=[img]))

    iii = p.image(image=[img], x=0, y=w, dw=w, dh=w, palette="Spectral9")#"Greys256")
    source = ColumnDataSource(data=dict(top=[0], bottom=[0], left=[0], right=[0]))
    q = p.quad('left', 'right', 'top', 'bottom', source=source, color=color, fill_alpha=0, line_width=3)
    
    return p, iii, q;

(unhover, i1, q1), (hover, i2, q2) = make_figure("pink",0), make_figure("orange",1)
figures.append([unhover, make_spacer(), hover])
iqs.append([
    (i1, q1),
    (i2, q2)
])

#(unhover_i, unhover_q), (hover_i, hover_q) = iqs
#unhover_i.data_source.data["image"][0] = data["original_img"]#data["rs"][i][0]
#hover_i.data_source.data["image"][0] = data["filter_img"] #data["rs"][i][1]

#unhover_q.data_source.data = data["rects"][i][0]
#hover_q.data_source.data = data["rects"][i][1]
        
    
#def on_click(b, new_image=True):
    #b.description = "Loading..."
    #update_figures(new_image=new_image)
    #b.description = "Next (Random) Image"
    
#b.on_click(on_click)

#def on_change(change):
    #if change['type'] == 'change' and change['name'] == 'value':
        #on_click(b, new_image=False)

#display(HBox([b]))
handle = show(layout(figures), notebook_handle=True)
# update_figures(handle)
#on_click(b)

