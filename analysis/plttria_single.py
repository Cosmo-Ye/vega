from getdist import loadMCSamples
from getdist import plots
import matplotlib.pyplot as plt
import sys

"""
    Usage: python python plttria.py <points> #<label>
    e.g. if points file is lya_auto_fid/lyaxlya_lyaxlyb_s1.txt then
    python plttria.py lya_auto_fid/lyaxlya_lyaxlyb_s1 #lya_auto_fid/file.paramnames
    
"""

plt.style.use('../perf.mplstyle')
chain_file = sys.argv[1]


print(chain_file)
#print(label)
s = loadMCSamples(str(chain_file),settings={'ignore_rows': 0.2})


samples = []
samples.append(s)
labels = ['Yuk3'] #['Fid.']


g = plots.get_subplot_plotter(subplot_size=1)

#g.settings.line_styles = ['-C0', '--C1', '-.C3', ':C2', '--C7', ':C5']
g.settings.line_styles = ['--C2', '-C0']
g.settings.linewidth = 4
g.settings.solid_colors = ['C2', 'C0']
g.settings.alpha_filled_add = 0.75
g.settings.title_limit = 1
g.settings.title_limit_fontsize = 14
g.settings.legend_fontsize = 28
g.settings.figure_legend_frame = False
#g.triangle_plot(s, legend_labels=[label], filled=True)
g.triangle_plot(samples, legend_labels=labels, filled=True)
g.export('corner.pdf')



