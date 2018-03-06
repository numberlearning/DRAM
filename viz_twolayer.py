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

from analysis_twolayer import classify_image, read_n, classify_imgs2

clear_output()
b = Button(description="Loading...", icon="arrow", width=400)
dropdown = Dropdown(
    options=['0', '1000', '2000', '3000', '4000', '5000', '10000', '20000', '30000', '40000', '50000', '60000', '70000', '80000', '90000', '100000', '110000', '120000', '130000', '140000', '150000', '160000', '170000', '180000', '190000', '200000', '250000', '300000', '400000', '500000', '600000', '700000', '800000', '900000', '1000000', '1100000', '1200000', '1300000', '1400000', '1500000', '3474000'],
    value='10000',
    description='Iteration:'
)

figures = list()
iqs = list()
charts = list()
curves = list()

def make_chart(i, j): # only need one chart, and there are no glimpses
    """
    i: glimpse number
    j: Outer-Layer if 0, Inner-Layer if 1
    """
#     source = ColumnDataSource(data=dict(data=))
#     bar2 = Bar(data=np.random.rand(10), title="Python Interpreters", plot_width=400, legend=False)
    name = "Outer-Layer" if j == 0 else "Inner-Layer"
    title = "%s Classification %d" % (name, (i + 1))
    p = figure(x_range=(-0.5, 10), y_range=(0, 1), width=200, height=250, tools="")
    
    m = 0.1
    source = ColumnDataSource(data=dict(color=["lime"] * 10, top=np.zeros(10), bottom=np.zeros(10), left=np.arange(10) + m - 0.5, right=np.arange(1, 11) - m - 0.5))
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
    j: Outer-Layer if 0, Inner-Layer if 1
    smol: if the image should be small
    """
    if smol:
        w = read_n
    else:
        w = 100
        
    name = "Outer-Layer" if j == 0 else "Inner-Layer"
    title = "%s " % (name)
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

# there is no i
(outer, i1, q1), (inner, i2, q2) = make_figure("pink", i, 0), make_figure("orange", i, 1)
outer_c, outer_cdata = make_chart(i, 0)
inner_c, inner_cdata = make_chart(i, 1)
figures.append([outer, outer_c, make_spacer(), inner, inner_c])
iqs.append([
    (i1, q1),
    (i2, q2)
])
charts.append([outer_cdata, inner_cdata])

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
    data = classify_image(int(dropdown.value), new_image)
    for i, f in enumerate(figures):
        
        outer, outer_c, spacer, inner, inner_c = f
        
        (outer_i, outer_q), (inner_i, inner_q) = iqs[i]
        outer_i.data_source.data["image"][0] = data["img"]#data["rs"][i][0]
        inner_i.data_source.data["image"][0] = data["img"] #data["rs"][i][1]
        
        outer_q.data_source.data = data["rects"][i][0]
        inner_q.data_source.data = data["rects"][i][1]
        
        outer_cdata, inner_cdata = charts[i]
        
        def colorify(ar):
            colors = []
            for x in ar:
                colors.append("lime" if x else "red")
            return colors
        
        clabel = colorify(data["label"])
                
        
        outer_cdata.data_source.data["top"] = data["classifications"][i][0]
        outer_cdata.data_source.data["color"] = clabel
        inner_cdata.data_source.data["top"] = data["classifications"][i][1]
        inner_cdata.data_source.data["color"] = clabel

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
