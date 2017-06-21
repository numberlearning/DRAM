*import analysis
*import numpy as np
*import matplotlib as mpl
*mpl.use('TkAgg')
*import matplotlib.pyplot as plt
*import plotly.plotly as py

*glimpses = 10
*output_size = 9
*a = np.zeros([glimpses, output_size, output_size + 1])

*print("graph.py")
*print("ALL-STEP", analysis.accuracy_stats(40, True))
*print("LAST-STEP", analysis.accuracy_stats(40, False))

*pred_distr_at_glimpses = analysis.accuracy_stats(40, False)
########################################################
#MATPLOTLIB
#################
*x = a[:, 0, :]
*plt.hist(x, normed=True, bins=output_size)
*plt.ylabel('Prediction Distribution for Input of 1 Blob')
*plt.show()


########################################################
#PLOTLY
#################
*colormaps_fit = plt.figure()
 
*num_plots = 9
 
*colormap = plt.cm.gist_ncar
*plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
 
*x = np.arrange(10)
*labels = []
*for i in range(1, num_plots + 1):
*    plt.plot(x, a[9, i-i, x])
*    labels.append(r'$y = %ix + %i$' % (i, 5*i))
 
*plot_url = py.plot_mpl(colormaps_fig, filename = 'mpl-colormaps')
