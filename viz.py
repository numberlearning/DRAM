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

import ipywidgets as widgets
from ipywidgets import *
from IPython.display import display, clear_output

import numpy as np

from bokeh.charts import Bar, Histogram

from analysis import classify_image, glimpses, read_n

clear_output()
b = Button(description="Loading...", icon="arrow", width=400)
dropdown = Dropdown(
    options=['0', '1000', '2000', '3000', '4000', '5000', '10000', '20000', '30000', '50000', '100000', '200000', '300000', '400000', '500000'],
    value='1000',
    description='Iteration:'
)

figures = list()
iqs = list()
charts = list()

def make_chart(i, j):
    """
    i: glimpse number
    j: Last-Step if 0, All-Step if 1
    """
#     source = ColumnDataSource(data=dict(data=))
#     bar2 = Bar(data=np.random.rand(10), title="Python Interpreters", plot_width=400, legend=False)
    name = "Last-Step" if j == 0 else "All-Step"
    title = "%s Classification %d" % (name, (i + 1))
    p = figure(x_range=(-0.5, 10), y_range=(0, 1), width=200, height=250, tools="")
    
    m = 0.1
    source = ColumnDataSource(data=dict(color=["lime"] * 10, top=np.zeros(10), bottom=np.zeros(10), left=np.arange(10) + m + 0.5, right=np.arange(1, 11) - m + 0.5))
    q = p.quad('left', 'right', 'top', 'bottom', source=source, color="color")


    p.xaxis.axis_label = 'True Class'
    p.yaxis.axis_label = 'Probability'
    
    p.toolbar.logo = None
    p.toolbar_location = None
    p.xaxis[0].ticker=FixedTicker(ticks=np.arange(10))

    
    return p, q

def make_spacer():
    p = figure(x_range=(-0.5, 10), y_range=(0, 1), width=50, height=250, tools="", border_fill_alpha=0, outline_line_alpha=0)
    p.toolbar.logo = None
    p.toolbar_location = None
    p.border_fill_alpha = 0
    p.outline_line_alpha = 0
    
    p.axis.visible = False
    p.grid.visible = False
    
    return p

def make_figure(color, i, j, smol=False):
    """
    color: attention window color
    i: glimpse number
    j: Last-Step if 0, All-Step if 1
    smol: if the image should be small
    """
    if smol:
        w = read_n
    else:
        w = 100
        

    name = "Last-Step" if j == 0 else "All-Step"
    title = "%s Glimpse %d" % (name, (i + 1))
    p = figure(x_range=(0, w), y_range=(w, 0), width=200, height=250, tools="", title=title)
    
    p.toolbar.logo = None
    p.toolbar_location = None
    p.axis.visible = False

    im = np.zeros((w, w))
    i_source = ColumnDataSource(data=dict(image=[im]))


    iii = p.image(image=[im], x=0, y=w, dw=w, dh=w, palette="Spectral9")#"Greys256")
    source = ColumnDataSource(data=dict(top=[0], bottom=[0], left=[0], right=[0]))
#     source = ColumnDataSource(data=dict(top=[d["top"]], bottom=[d["bottom"]], left=[d["left"]], right=[d["right"]]))
    q = p.quad('left', 'right', 'top', 'bottom', source=source, color=color, fill_alpha=0, line_width=3)
    
        
    callback = CustomJS(code="""
    console.log(cb_data);
    if (IPython.notebook.kernel !== undefined) {
        var kernel = IPython.notebook.kernel;
        var i = %d;
        var j = %d;
        if (!this.hovered) {
            cmd = "hover(" + i + ", " + j + ")";
            kernel.execute(cmd, {}, {});
            this.hovered = true;
        }
        
        var that = this;
        
        
        document.querySelectorAll(".bk-plot-layout.bk-layout-fixed").forEach(function(x) {
            x.onmouseleave = function() {
                if (!that.hovered) {
                    return;
                }
                that.hovered = false;
                cmd = "unhover(" + i + ", " + j + ")";
                kernel.execute(cmd, {}, {});
            }
        })
        
    }
    """ % (i, j))
    p.add_tools(HoverTool(tooltips=None, callback=callback, renderers=[iii, q]))
    
    
    return p, iii, q;

for i in range(glimpses):
    #if i % 10 == 0:
    (machine, i1, q1), (human, i2, q2) = make_figure("pink", i, 0), make_figure("orange", i, 1)
    machine_c, machine_cdata = make_chart(i, 0)
    human_c, human_cdata = make_chart(i, 0)
    figures.append([machine, machine_c, make_spacer(), human, human_c])
    iqs.append([
        (i1, q1),
        (i2, q2)
    ])
    charts.append([machine_cdata, human_cdata])
        
data = None
    
def hover(i, j):
    """
    Show attention window image when figure is hovered over.
    i: glimpse number
    j: Last-Step or All-Step
    iqs: list of images and attention windows
    """
    iqs[i][j][0].data_source.data["image"][0] = data["rs"][i][j]
    iqs[i][j][1].data_source.data = dict(top=[0], bottom=[0], left=[0], right=[0])
    push_notebook(handle=handle)


def unhover(i, j):
    """
    Show figure image when figure is unhovered.
    i: glimpse number
    j: Last-Step or All-Step
    iqs: list of images and attention windows
    """
    iqs[i][j][0].data_source.data["image"][0] = data["img"]
    iqs[i][j][1].data_source.data = data["rects"][i][j]
    push_notebook(handle=handle)
    
    
def update_figures(handle, new_image=True):
    global data
    data = classify_image(int(dropdown.value), new_image=new_image)
    for i, f in enumerate(figures):
        
        machine, machine_c, spacer, human, human_c = f
        
        (machine_i, machine_q), (human_i, human_q) = iqs[i]
        machine_i.data_source.data["image"][0] = data["img"]#data["rs"][i][0]
        human_i.data_source.data["image"][0] = data["img"] #data["rs"][i][1]
        
        machine_q.data_source.data = data["rects"][i][0]
        human_q.data_source.data = data["rects"][i][1]
        
        machine_cdata, human_cdata = charts[i]
        
        def colorify(ar):
            colors = []
            for x in ar:
                colors.append("lime" if x else "red")
            return colors
        
        print('data["label"]: ')
        print(data["label"])
        clabel = colorify(data["label"])
                
        
        machine_cdata.data_source.data["top"] = data["classifications"][i][0][0]

        print('data["classifications"][i][0][0]: ')
        print(data["classifications"][i][0][0])

        print('data["rects"][i][j]: ')
        # for glimpse in range(glimpses):
        print(data["rects"][i][0])

        machine_cdata.data_source.data["color"] = clabel
        
        human_cdata.data_source.data["top"] = data["classifications"][i][1][0]
        human_cdata.data_source.data["color"] = clabel

                
    push_notebook(handle=handle)
    
def on_click(b, new_image=True):
    b.description = "Loading..."
    update_figures(handle, new_image=new_image)
    b.description = "Next (Random) Image"
    
b.on_click(on_click)

def on_change(change):
    if change['type'] == 'change' and change['name'] == 'value':
        on_click(b, new_image=False)

dropdown.observe(on_change)
display(HBox([b, dropdown]))
handle = show(layout(figures), notebook_handle=True)
# update_figures(handle)
on_click(b)
