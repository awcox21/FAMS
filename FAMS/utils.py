from collections import OrderedDict
from math import factorial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from pandas import DataFrame, Series
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import brentq
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from scipy.stats import norm


def dist_median(xs, ys, plot=False, ax=None, _spline=None):
    return dist_percentile(xs, ys, 50, plot, ax, _spline)


def dist_percentile(xs, ys, percent, plot=False, ax=None, _spline=None,
                    cmap=None):
    min_, max_ = min(xs), max(xs)
    if not _spline:
        spline = InterpolatedUnivariateSpline(xs, ys)
    else:
        spline = _spline
    total = spline.integral(min_, max_)

    def fun(_x):
        area = total - spline.integral(min_, _x)
        return area / total - percent / 100

    x_percent = brentq(fun, min_, max_)
    if plot:
        if ax is None:
            plt.figure()
            ax = plt.gca()
        ax.plot(xs, ys)
        count = 0
        for count, x in enumerate(xs):
            if x >= x_percent:
                break
        ax.fill_between(
            xs[:count], [0 for _ in range(count)], ys[:count],
            label='_nolegend_', alpha=0.4, color=cmap(0))
    return x_percent, spline


def flatten(nested):
    """
    Flatten list of lists

    Parameters
    ----------
    nested : List

    Returns
    -------
    flattened : List
    """
    flattened = list()
    for list_ in nested:
        list_ = normalize(list_)
        flattened.extend(list_)
    return normalize(flattened)


def get_colors(num, cmap=None, reverse=False, min_=None, max_=None):
    """
    Get a discrete number of colors from a colormap

    Parameters
    ----------
    num : int
    cmap : ListedColormap, optional
    reverse : bool, optional
    min_ : Union[int, float]
        Float from 0.0 to 1.0 or int from 0 to 255
    max_ : Union[int, float]
        Float from 0.0 to 1.0 or int from 0 to 255

    Returns
    -------
    colors : List[tuple]
    """
    if not min_:
        min_ = 0
    if not max_:
        max_ = 1
    cmap = cm.get_cmap(cmap)
    colors = [cmap(_value) for _value in np.linspace(min_, max_, num)]
    if reverse:
        colors = reversed(colors)
    return colors


def get_kde(samples, sample_weight=None, bandwidth=None, plot=False, cmap=None,
            label=None, plot_samples=True, plot_dists=True, linestyle=None):
    """
    Fit kernel density estimate to sample scores

    Parameters
    ----------
    samples
    sample_weight
    bandwidth
    plot
    cmap
    label
    plot_samples
    plot_dists

    Returns
    -------
    kde
    """
    if sample_weight:
        if len(samples) != len(sample_weight):
            raise ValueError('If sample weights provided, '
                             'scores and weights must be same size')
        kwargs = {'sample_weight': sample_weight}
    else:
        kwargs = dict()
    samples = np.array(samples)
    if bandwidth is None:
        params = {'bandwidth': 5 * np.logspace(-2, 1, 20)}
        grid = GridSearchCV(KernelDensity(), params, cv=len(samples))
        grid.fit(samples[:, None], **kwargs)
        kde = grid.best_estimator_
    else:
        kde = KernelDensity(bandwidth)
        kde.fit(samples[:, None], **kwargs)
    if plot:
        if isinstance(cmap, tuple):
            color = cmap
        else:
            cmap_ = cm.get_cmap(cmap)
            color = cmap_(0.0)
        kwargs = {'color': color}
        std = 2 * kde.bandwidth ** 0.5
        xs = np.linspace(min(samples) - std, max(samples) + std, 1000)
        ys = np.exp(kde.score_samples(xs[:, None]))
        if label:
            kwargs['label'] = label
        if linestyle is not None:
            kwargs['linestyle'] = linestyle
        plt.plot(xs, ys, **kwargs)
        max_y = max(ys)
        plt.fill_between(xs, [0 for _ in xs], ys, color=color, alpha=0.3)
        if plot_samples or plot_dists:
            divisions = max(list(samples).count(_) for _ in samples) + 9
            offset = max(ys) / divisions
            plotted = dict()
            sigma = kde.bandwidth ** 0.5
            for sample in samples:
                try:
                    plotted[sample] += 1
                except KeyError:
                    plotted[sample] = 0
                if plot_samples:
                    plt.scatter((sample,), (offset * plotted[sample],),
                                color=color, marker='x')
                if plot_dists:
                    xs = np.linspace(sample - 3 * sigma, sample + 3 * sigma,
                                     1000)
                    ys = norm.pdf(xs, sample, sigma) * max_y / len(samples)
                    plt.plot(xs, ys, color=color, label='_nolegend_',
                             alpha=0.5)
        plt.yticks(list())
        ax = plt.gca()
        for side in ['left', 'right', 'top']:
            ax.spines[side].set_visible(False)
    return kde


def get_kde_percentile(samples, kde, percent, num_samples=1000, ax=None,
                       plot=False, _spline=None, cmap=None):
    """
    Get quartile based on KDE distribution

    Parameters
    ----------
    samples
    kde
    percent
    num_samples
    plot
    _spline
    cmap

    Returns
    -------
    x_percent
    spline
    """
    var = 3 * kde.bandwidth ** 0.5
    min_, max_ = min(samples) - var, max(samples) + var
    xs = np.linspace(min_, max_, num_samples)
    ys = np.exp(kde.score_samples(xs[:, None]))
    return dist_percentile(xs, ys, percent, plot, ax, _spline, cmap)


def normalize(list_):
    """
    Normalize values by sum of list

    Parameters
    ----------
    list_ : List[float]

    Returns
    -------
    normalized : List[float]
    """
    return [_i / sum(list_) for _i in list_]


def num_combinations(size, start=1, stop=None):
    """
    Number of non-ordered combinations

    Parameters
    ----------
    size : int
    start : int, optional
        Default value is 1
    stop : int, optional
        Default value is same as size. If start and stop are the same
        value, returns number of permutations choosing that value.

    Raises
    ------
    ValueError
        When stop is less than start

    Returns
    -------
    num : int
    """
    num = 0
    if stop is None:
        stop = size + 1
    else:
        stop += 1
    if stop <= start:
        raise ValueError('Stop must be at >= start or None')
    for i in range(start, stop):
        num += int(
            factorial(size) / factorial(i) / factorial(size - i))
    return num


def num_permutations(size, start=1, stop=None):
    """
    Number of ordered permutations of options

    Parameters
    ----------
    size : int
    start : int, optional
        Default value is 1
    stop : int, optional
        Default value is same as size. If start and stop are the same
        value, returns number of permutations choosing that value.

    Raises
    ------
    ValueError
        When stop is less than start

    Returns
    -------
    num : int
    """
    num = 0
    if stop is None:
        stop = size + 1
    else:
        stop += 1
    if stop <= start:
        raise ValueError('Stop must be at >= start or None')
    for i in range(start, stop):
        num += int(factorial(size) / factorial(size - i))
    return num


def oec(scores, weights=None):
    """
    Overall evaluation criterion method

    Notes
    -----
    Code not obsolete, but method is

    Parameters
    ----------
    scores : List[Series]
    weights : List[float]

    Returns
    -------
    oecs : DataFrame
    """
    if not weights:
        weights = [1 / len(scores) for _ in scores]
    orders = set(scores[0].order)
    for scores_ in scores[1:]:
        if orders != set(scores_.order):
            raise ValueError('Different sets of orders')
    orders = sorted(orders)
    scores = [_scores.sort_values('order') for _scores in scores]
    scores_ = [_scores.score.values[:, None] for _scores in scores]
    scores_ = np.hstack([_scores / max(_scores) for _scores in scores_])
    oecs = np.dot(scores_, weights)
    oecs = [(_order, _oec) for _order, _oec in zip(orders, oecs)]
    oecs = DataFrame(oecs, columns=['order', 'oec'])
    oecs = oecs.sort_values('oec', ascending=False)
    oecs.index = range(1, len(oecs) + 1)
    oecs['cumpercentage'] = (oecs['oec'].cumsum() / oecs['oec'].sum() * 100)
    return oecs


def plot_proportion(x, scores, width=0.8, new_fig=False, models=None,
                    cmap=None, legend=False):
    """
    Plot proportions in a stacked bar chart at location x with given
    width.

    Parameters
    ----------
    x : float
    scores : Dict[str: float]
    width : float, optional
    new_fig : bool, optional, optional
    models : List[Model], optional
    cmap
    legend : bool, optional
    """
    if new_fig:
        plt.figure()
    if cmap in (cm.gray, cm.gray_r):
        colors = get_colors(len(scores), cmap, min_=80/256, max_=176/256)
    else:
        colors = get_colors(len(scores), cmap, min_=48/256, max_=208/256)
    bottom = 0
    for c, key in zip(colors, sorted(scores)):
        score = scores[key]
        if models:
            for model in models:
                if model.id == key:
                    break
            else:
                raise AttributeError('No matching model ID')
            kwargs = {'label': str(model)}
        else:
            kwargs = {'label': 'Model {}'.format(key)}
        bar = plt.bar(x, score, width, bottom, color=c, **kwargs)
        ax = plt.gca()
        for rect in bar:
            if rect.get_height() > 0.03:
                ax.text(rect.get_x() + rect.get_width() / 2.0,
                        rect.get_y() + score / 2.0,
                        '{}: {:f}'.format(key, score), ha='center',
                        va='center')
        bottom += score
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    if legend:
        plt.legend(by_label.values(), by_label.keys(),
                   bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left',
                   mode='expand', ncol=len(scores))


def remove_outliers(
        data, column=None, num_std=2.5, iqr_scale=1.5, low=True, high=True):
    """
    Remove outliers using iterative Z-score and IQR methods

    Parameters
    ----------
    data : Union[Series, DataFrame]
    column : str, optional
    num_std : float, optional
    iqr_scale : float, optional
    low : bool, optional
    high : bool, optional

    Returns
    -------
    data : Union[Series, DataFrame]
    """
    if isinstance(data, Series):

        def _mean(_data):
            return _data.mean()

        def _std(_data):
            return _data.std()

        def _quantile(_data, q):
            return _data.quantile(q)

        def _trim(_data, values, lower):
            clear = False
            if lower:
                for value in values:
                    if len(_data[_data < value]) > 0:
                        _data = _data[_data > value]
                        break
                else:
                    clear = True
            else:  # higher
                for value in values:
                    if len(_data[_data > value]) > 0:
                        _data = _data[_data < value]
                        break
                else:
                    clear = True
            return _data, clear

    elif isinstance(data, DataFrame):
        if column is None:
            raise ValueError('Column must be defined for DataFrame')

        def _mean(_data):
            return _data[column].mean()

        def _std(_data):
            return _data[column].std()

        def _quantile(_data, q):
            return _data[column].quantile(q)

        def _trim(_data, values, lower):
            clear = False
            if lower:
                for value in values:
                    if len(_data[column][_data[column] < value]) > 0:
                        _data = _data[column][_data[column] > value]
                        break
                else:
                    clear = True
            else:  # higher
                for value in values:
                    if len(_data[column][_data[column] > value]) > 0:
                        _data = _data[column][_data[column] < value]
                        break
            return _data, clear

    else:
        raise TypeError('Data must be Series or DataFrame')
    num_required = 0
    if low:
        num_required += 1
    if high:
        num_required += 1
    while True:
        num_clear = 0
        mean, std = _mean(data), _std(data)
        q25, q75 = _quantile(data, 0.25), _quantile(data, 0.75)
        iqr = q75 - q25
        cut_off = iqr_scale * iqr
        if low:
            data, clear = _trim(
                data, [mean - num_std * std, q25 - cut_off], True)
            if clear:
                num_clear += 1
        if high:
            data, clear = _trim(
                data, [mean + num_std * std, q75 + cut_off], False)
            if clear:
                num_clear += 1
        if num_clear == num_required:
            break
    return data
