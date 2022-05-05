import random
from itertools import cycle

import matplotlib.pyplot as plt
from matplotlib.cm import gray, gray_r, viridis
import numpy as np
import pandas as pd

from FAMS.model_rankings import Model, Order, Ranking
from FAMS.utils import (
    plot_proportion, get_kde, get_kde_percentile, num_permutations)

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

# plt.ion()
# plt.show()

random.seed(0)

num_models, num_orders = 4, 1
print('Permutations: {}'.format(num_permutations(num_models)))
fig, ax = plt.subplots()
cases = ('all-increasing', 'no-scope', 'fixed-scope', 'realistic')
""" NOTE: Alternate these `times` options below to see compare relative notional run-times """
# times = np.linspace(100, 101, num_models)  # minimal time variation
# times = np.logspace(0, 2, num_models)  # exponential time variation
times = np.linspace(1, 100, num_models)  # linear time variation
for count, case in enumerate(cases):
    models = [Model() for _ in range(num_models)]
    time_column = 'Time'
    time_columns = time_column,
    for model, time in zip(models, times):
        model.data = pd.DataFrame({time_column: [time, time],
                                   'Input': [1.0, 1.0]})
    input_columns = 'Input',

    resolution = Ranking(models, 'Resolution')
    resolution.add_order(Order([[_model] for _model in models]))  # simple order
    # for order in Order.randomly_generate_orders(models, num_orders):
    #     resolution.add_order(order)

    abstraction = Ranking(models, 'Abstraction')
    abstraction.add_order(Order([[_model] for _model in models]))  # simple order
    # for order in Order.randomly_generate_orders(models, num_orders):
    # #     abstraction.add_order(order)

    scope = Ranking(models, 'Scope')
    if case == 'fixed-scope':
        scope.add_order(Order([models]))  # fixed scope
    elif case == 'all-increasing':
        scope.add_order(Order([[_model] for _model in models]))  # simple order
    elif case == 'realistic':
        scope.add_order(Order(reversed([[_model] for _model in models])))
    # for order in generate_orders(models):  # variable scope
    #     scope.add_order(order)

    if case == 'no-scope':
        combined = Ranking.combine([resolution, abstraction], name='notional')
    else:
        combined = Ranking.combine([resolution, abstraction, scope],
                                   name='notional')

    # case = 'no-scope'
    # case = 'fixed-scope'
    # case = 'all-increasing'
    # case = 'all-increasing-log-times'
    # case = 'all-increasing-lin-times'
    # case = 'all-increasing-similar-times'
    # case = 'realistic'
    # case = 'realistic-log-times'
    # case = 'realistic-lin-times'
    # case = 'realistic-similar-times'

    if plot:
        combined_models = combined.plot_scores(
            cdf=cdf, cmap=mono_cmap, title=False, legend=False, use_ids=True)
        if tight:
            plt.tight_layout()
        if save:
            plt.savefig('notional-dists-{}.{}'.format(case, save))

    """ Notional probs """
    prob_levels = [combined.prob_level(_i) for _i in range(1, len(models) + 1)]
    if plot:
        plt.figure()
        for i, scores in enumerate(prob_levels):
            plot_proportion(i, scores, cmap=cmap, models=models, legend=False)
        plt.xticks(range(4),
                   ['P(1$^{st}$)', 'P(2$^{nd}$)', 'P(3$^{rd}$)', 'P(last)'])
        if tight:
            plt.tight_layout()
        if save:
            plt.savefig('notional-probs-{}.{}'.format(case, save))

    scores = combined.multifidelity_scoring(
        False, threshold, cmap, freq_only, dark_background)
    off = 0.01
    single_labels, single_ys = list(), list()
    multi_labels, multi_ys = list(), list()
    extra_labels, extra_ys = list(), list()
    for _, order, score, cumperc in scores.itertuples():
        if cumperc <= threshold and len(single_ys) + len(multi_ys) <= 5:
            if not order.count(','):
                single_labels.append(order)
                single_ys.append(score)
            else:
                multi_labels.append(order)
                multi_ys.append(score)
        else:
            if not order.count(','):
                if not any(abs(score - _) < off for _ in single_ys):
                    single_labels.append(order)
                    single_ys.append(score)
                elif not any(abs(score - _) < off for _ in multi_ys):
                    extra_labels.append(order)
                    extra_ys.append(score)
    xs = [count for _ in single_ys]
    color = mono_cmap(0.0)
    if not count:
        label = 'Single Model'
    else:
        label = '_nolegend_'
    ax.scatter(xs, single_ys, color=color, label=label)
    for x, y, label in zip(xs, single_ys, single_labels):
        ax.annotate(label, (x, y), xytext=(x + off, y + off))
    xs = [count + 1 / 6 for _ in extra_ys]
    ax.scatter(xs, extra_ys, color=color, label='_nolegend_')
    for x, y, label in zip(xs, extra_ys, extra_labels):
        ax.annotate(label, (x, y), xytext=(x + off, y + off))
    xs = [count + 1 / 3 for _ in multi_ys]
    color = mono_cmap(50/256)
    if not count:
        label = 'Multifidelity'
    else:
        label = '_nolegend_'
    ax.scatter(xs, multi_ys, color=color, label=label, marker='X')
    for x, y, label in zip(xs, multi_ys, multi_labels):
        ax.annotate(label, (x, y), xytext=(x + off, y + off))
ymin, ymax = ax.get_ylim()
for i in range(count):
    ax.vlines(i + 3 / 4, ymin, ymax)
xticks = list()
for i in range(count + 1):
    xticks.append(i + 1 / 4)
ax.set_ylim(ymin, ymax)
xmin, _ = ax.get_xlim()
ax.set_xlim(xmin, count + 2 / 3)
ax.set_xticks(xticks)
ax.set_xticklabels([_.title() for _ in cases])
ax.legend(loc='lower right')
ax.set_ylabel('Fidelity Score')
fig.tight_layout()
if save:
    fig.savefig('notional-fidelity-scores.{}'.format(save))
if tight:
    plt.tight_layout()
if save:
    plt.savefig('notional-multi-{}.{}'.format(case, save))

""" Notional Costs """
if save:
    save_ = '-{}.{}'.format(case, save)
else:
    save_ = save
costs = combined.estimate_cost(
    time_columns, plot=plot, cmap=cmap, use_ids=False, tight=tight, save=save_)
crs = combined.estimate_cost_ratios(
    input_columns, time_columns, plot=plot, cmap=cmap)
dominant = combined.multiattribute_scoring(
    input_columns, time_columns, plot=plot, cmap=mono_cmap, plot_single=True)
if tight:
    plt.tight_layout()
if save:
    plt.savefig('notional-pareto-{}.{}'.format(case, save))

"""
Recommended that if you can't decide whether two models have a rank order or 
are equivalent just go ahead and include both options as separate opinions.
The resulting scores will be closer together but still show a difference.
"""

plt.show()
