import analysis
import DRAMcopy10-nli_classification as DRAM
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import plotly.plotly as py

print("ALL-STEP", accuracy_stats(300000, True))
print("LAST-STEP", accuracy_stats(300000, False))


