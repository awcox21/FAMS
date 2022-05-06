import matplotlib.pyplot as plt
from matplotlib.cm import gray, gray_r, viridis
import random

from FAMS.model_rankings import Model

fig_width = 6.4  # matplotlib default
fig_ar = 4 / 3  # default is 4 / 3
fig_height = fig_width / fig_ar
plt.rc('font', family='Times New Roman', size=12)
plt.rc('figure', figsize=(fig_width, fig_height))
plt.rc('savefig', dpi=600, transparent=True)

cdf = False
cmap, plot = viridis, True
# dark_background, tight, save = True, True, 'svg'  # for presentation
dark_background, tight, save = False, False, False  # for running and testing
# dark_background, tight, save = False, True, 'pdf'  # for document
threshold, freq_only = 80, False
if dark_background:
    mono_cmap = gray_r
    plt.rc(('xtick', 'ytick'), color='w')
    plt.rc('text', color='w')
    plt.rc('axes', edgecolor='w', labelcolor='w')
    plt.rc('legend', framealpha=0.)
else:
    mono_cmap = gray

random.seed(2)  # for randomly generating orders if needed

num_models = 4

models = [Model() for _ in range(num_models)]
