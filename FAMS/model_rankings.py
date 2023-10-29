from itertools import combinations, permutations, cycle
from math import sqrt, log, ceil
from operator import attrgetter, itemgetter
from random import shuffle
from warnings import warn

import matplotlib.markers
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from matplotlib import cm
from matplotlib.ticker import PercentFormatter
from numpy import linspace
from pandas import DataFrame, Series
from scipy.interpolate import InterpolatedUnivariateSpline, griddata
from scipy.optimize import brentq
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from time import time
from adjustText import adjust_text

from .utils import (
    dist_median, flatten, get_colors, get_kde,
    get_kde_percentile, normalize, remove_outliers, num_permutations,
    pareto_front, plot_proportion, ordinal)

markers = matplotlib.markers.MarkerStyle().markers
included_markers = ['point', 'x', 'star', 'octagon', 'pentagon', 'square',
                    'diamond']
directions = ['down', 'left', 'right', 'up']
for direction in directions:
    included_markers.append('tri_{}'.format(direction))
markers = [(key, markers[key]) for key in markers if markers[key] in
           included_markers]
markers.sort(key=itemgetter(1))
markers = [key for key, _ in markers]


class Item(object):
    def __init__(self, name=None, id_=None):
        self._name = name
        self._id = id_

    def __str__(self):
        if not self._name:
            if self.id is None:
                return 'Item'
            else:
                return 'Item {}'.format(self.id)
        else:
            return self._name

    def __repr__(self):
        return "Item(name='{}', id_={})".format(self.name, self.id)
    
    @property
    def id(self):
        if self._id is None:
            raise AttributeError
        return self._id

    @id.setter
    def id(self, id_):
        self._id = id_
        
    @property
    def name(self):
        return str(self)


class Model(Item):
    def __init__(self, name=None, id_=None, data=None):
        """
        Model data object

        Parameters
        ----------
        name : str, optional
        id_ : int, optional
        data : DataFrame, optional
        """
        super().__init__(name, id_)
        self._data = data
        self._gprs = dict()

    def __str__(self):
        if not self._name:
            if self.id is None:
                return 'Model'
            else:
                return 'Model {}'.format(self.id)
        else:
            return self._name

    def __repr__(self):
        return "Model(name='{}', id_={})".format(self.name, self.id)

    @property
    def data(self):
        if self._data is None:
            raise ValueError('No data assigned yet')
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    def filter(self, input_keys, value_sets):
        """
        Return a subset of model data based on input values

        Parameters
        ----------
        input_keys : iterable[str]
        value_sets : iterable[iterable[float]]

        Returns
        -------
        filtered_data : DataFrame
        """
        if isinstance(input_keys, str):
            input_keys = [input_keys]
        else:
            try:
                iter(input_keys)
            except TypeError:
                input_keys = [input_keys]
        if all(isinstance(_value, float) for _value in value_sets):
            value_sets = [value_sets]
        else:
            value_sets_ = list()
            for values in value_sets:
                try:
                    iter(values)
                except TypeError:
                    values = [values]
                value_sets_.append(values)
            value_sets = list(value_sets_)
        data = self.data.copy()
        for key, values in zip(input_keys, value_sets):
            data = data[data[key].isin(values)]
        return data

    def get_gpr(self, input_columns, output_columns, kernel=None,
                num_test=3, _io=None, _plot=False):
        """
        Single-model Gaussian Process Regressors

        Parameters
        ----------
        input_columns : List[str]
        output_columns : List[str]
        kernel : kernel object, optional
            The kernel specifying the covariance function of the GP. If
            None is passed, the kernel "1.0 * RBF(1.0)" is used as
            default. Note that the kernel's hyperparameters are
            optimized during fitting. Default value is ConstantKernel * RBF.
        num_test : int, optional
            Number of length scale values in logspace to check for global
            max log marginal likelihood. Default value is 3.
        _io
        _plot

        Returns
        -------
        gpr : GaussianProcessRegressor
        """
        input_columns_ = ', '.join(sorted(input_columns))
        output_columns_ = ', '.join(sorted(output_columns))
        try:
            return self._gprs[input_columns_, output_columns_]
        except KeyError:
            if _plot:
                raise KeyError('Invalid GPR key')
        if _io:
            inputs, outputs = _io
        else:
            inputs, outputs = self.map_io(input_columns, output_columns)
        if not kernel:
            kernel = ConstantKernel() * RBF()
        gprs = dict()
        for value in np.logspace(-1, 2, num_test):
            kernel_ = clone(kernel)
            for key in kernel.get_params():
                if 'length_scale' in key and 'bounds' not in key:
                    kernel_.set_params(**{key: value})
            gp = GaussianProcessRegressor(kernel_)
            gp.fit(inputs, outputs)
            lml = gp.log_marginal_likelihood()
            if lml not in gprs:
                gprs[lml] = gp
        regressor = gprs[max(gprs)]
        self._gprs[input_columns_, output_columns_] = regressor
        return regressor

    def get_shared(self, other, input_columns, output_columns,
                   remove_duplicates=True):
        """
        Retrieve or interpolate (as possible) shared responses

        Notes
        -----
        Primarily for correlation calculations

        Parameters
        ----------
        other : Model
        input_columns : iterable[str]
            Column names for input values
        output_columns : iterable[str]
            Column names for output values
        remove_duplicates : bool, optional

        Returns
        -------
        inputs, (outputs, outputs_) : array[float]
            Inputs and outputs of model and other
        """
        inputs, outputs = self.map_io(
            input_columns, output_columns, remove_duplicates)
        inputs_, outputs_ = other.map_io(
            input_columns, output_columns, remove_duplicates)
        try:
            if np.allclose(inputs, inputs_):
                return inputs, (outputs, outputs_)
        except ValueError:  # not the same shape
            pass
        _bounds = [(min(_col), max(_col)) for _col in inputs.T]
        bounds = list()
        for (min_, max_), col in zip(_bounds, inputs_.T):  # no extrapolation
            if min(col) > max_:
                raise ValueError('Range mismatch: {}, {}'.format(self, other))
            elif min_ <= min(col) <= max_:
                min_ = min(col)
            if max(col) < min_:
                raise ValueError('Range mismatch: {}, {}'.format(self, other))
            elif min_ <= max(col) <= max_:
                max_ = max(col)
            bounds.append((min_, max_))
        for i, (min_, max_) in enumerate(bounds):
            min_filter = inputs[:, i] >= min_
            max_filter = inputs[:, i] <= max_
            inputs = inputs[min_filter & max_filter]
            outputs = outputs[min_filter & max_filter]
            min_filter_ = inputs_[:, i] >= min_
            max_filter_ = inputs_[:, i] <= max_
            inputs_ = inputs_[min_filter_ & max_filter_]
            outputs_ = outputs_[min_filter_ & max_filter_]
        values, values_ = dict(), dict()
        for inp, out in zip(inputs, outputs):  # for self
            values[tuple(inp)] = out
        for inp, out in zip(inputs_, outputs_):  # for other
            values_[tuple(inp)] = out
        diff_ = set(values).difference(values_)
        if diff_:
            predictions_ = griddata(inputs_, outputs_, np.array(list(diff_)),
                                    'linear', rescale=False)
            for value, prediction in zip(diff_, predictions_):
                values_[value] = prediction
        diff = set(values_).difference(values)
        if diff:
            predictions = griddata(inputs, outputs, np.array(list(diff)),
                                   'linear', rescale=False)
            for value, prediction in zip(diff, predictions):
                values[value] = prediction
        shared_outputs, shared_outputs_ = list(), list()
        try:
            assert sorted(values) == sorted(values_)
        except AssertionError:
            raise ValueError('Shared input mis-match')
        for inp in sorted(values):
            shared_outputs.append(values[inp])
            shared_outputs_.append(values_[inp])
        outputs, outputs_ = np.array(shared_outputs), np.array(shared_outputs_)
        return inputs, (outputs, outputs_)

    def map_io(self, input_columns, output_columns, remove_duplicates=True):
        """
        Map the column names of inputs and outputs to dictionaries of
        values

        Notes
        -----
        Output array is 1-D if only one response
        Input array is always 2-D
        This is for working with scikit-learn

        Parameters
        ----------
        input_columns : iterable[str]
            Column names for input values
        output_columns : iterable[str]
            Column names for output values
        remove_duplicates : bool, optional

        Returns
        -------
        inputs, outputs : array[float]
        """
        input_columns = list(input_columns)
        output_columns = list(output_columns)
        try:
            inputs = self.data[input_columns].values
        except KeyError:
            raise KeyError('Invalid input columns')
        else:
            if remove_duplicates:
                _inputs = inputs.copy()
                x = tuple(_inputs[0, :])
                inputs = [x]
                xs = {x}
                ignore = set()
                for i, x in enumerate(_inputs[1:]):
                    x = tuple(x)
                    if x in xs:
                        ignore.add(i + 1)
                    else:
                        xs.add(x)
                        inputs.append(x)
                if ignore:
                    inputs = np.array(inputs)
                else:
                    inputs = _inputs
            else:
                ignore = set()
            if not np.array_equal(np.atleast_2d(inputs), inputs):
                inputs = np.atleast_2d(inputs).T
        try:
            outputs = self.data[output_columns].values
        except KeyError:
            raise KeyError('Invalid output columns')
        else:
            if remove_duplicates:
                if ignore:
                    _outputs = outputs.copy()
                    outputs = list()
                    for i in range(len(_outputs)):
                        if i in ignore:
                            pass
                        else:
                            outputs.append(_outputs[i, :])
                    outputs = np.array(outputs)
            if len(output_columns) == 1:
                outputs = outputs.T[0]
            else:
                outputs = outputs
        return inputs, outputs

    def plot_gpr(self, input_columns, output_columns, num_samples=1000,
                 color=None, marker=None, normalize_=None):
        """
        Plots Gaussian process regression, showing mean and confidence
        bands

        Parameters
        ----------
        input_columns
        output_columns
        num_samples : float, optional
        color
        marker
        normalize_
        """
        gpr = self.get_gpr(input_columns, output_columns, _plot=True)
        kwargs = {'label': self.name}
        if color:
            kwargs['color'] = color
        if marker:
            kwargs['marker'] = marker
        ys = gpr.y_train_.copy()
        if normalize_:
            ys = (ys - normalize_) / normalize_ * 100
        plt.scatter(gpr.X_train_, ys, **kwargs)
        kwargs['label'] = '_nolegend_'
        try:
            kwargs.pop('marker')
        except KeyError:
            pass
        xs = np.linspace(min(gpr.X_train_), max(gpr.X_train_), num_samples)
        y_mean, y_std = gpr.predict(xs[:, None], return_std=True)
        if normalize_:
            y_mean = (y_mean - normalize_) / normalize_ * 100
            y_std /= normalize_
        plt.plot(xs, y_mean, **kwargs)
        plt.fill_between(
            xs, y_mean - y_std, y_mean + y_std, alpha=0.3)
        if normalize_:
            plt.ylabel('% Error from Baseline')

    def sort(self, key):
        """
        Sort values in-place by column title(s)

        Parameters
        ----------
        key : str, List[str]
        """
        if isinstance(key, str):
            key = [key]
        self._data = self.data.sort_values(key)
        

class Technology(Item):
    pass


class RankedOrder(object):
    def __call__(self):
        raise NotImplementedError('Subclass-specific method')

    def __repr__(self):
        rank = self.rank()
        max_ = 0
        for _, ids in rank:
            max_ = max([max_, len(str(ids))])
        len_ = max_ + 2
        str_ = '|{:^20}|{:^{len_}}|\n'.format('Score', 'IDs', len_=len_)
        for score, ids in self.rank():
            str_ += ('|{:^20}|{:^{len_}}|\n'.format(
                round(score, 16), str(ids), len_=len_))
        return str_

    def rank(self):
        """
        Gets the integer ID combined ranking

        Returns
        -------
        ranking : List[List[int]]
        """
        ranking_ = self()
        ids = sorted(ranking_, key=lambda x: ranking_[x])
        ranking, current = list(), list()
        id_ = None
        for id_ in ids:
            if not current:
                current.append(id_)
            else:
                if ranking_[current[-1]] == ranking_[id_]:
                    current.append(id_)
                else:
                    ranking.append((ranking_[current[-1]], current))
                    current = [id_]
        ranking.append((ranking_[id_], current))
        return ranking


class Order(RankedOrder):
    def __init__(self, order):
        """
        Parameters
        ----------
        order : List[List]
            List of lists of items. Sub-lists represent items deemed
            equivalent.
        """
        order = list(order)
        try:
            assert all(isinstance(_group, list) for _group in order)
        except AssertionError:
            raise ValueError('Invalid order list')
        self.order = list(order)

    def __call__(self):
        """
        Returns dictionary of normalized scores

        Returns
        -------
        order : Dict[int: float]
        """
        x = 1.0 / len(self.order)
        scores = [(_i + 1) * x * len(_grp) for _i, _grp in
                  enumerate(self.order)]
        order_ = dict()
        for score, group in zip(scores, self.order):
            for item in group:
                order_[item.id] = score / len(group) / sum(scores)
        return order_

    def combine(self, others):
        raise NotImplementedError('Subclass-specific method')

    @property
    def items(self):
        """
        All items in order, sorted by ID

        Returns
        -------
        items : List
        """
        items = [_i for _j in self.order for _i in _j]
        items.sort(key=attrgetter('id'))
        return items

    @classmethod
    def randomly_generate_orders(cls, items, num_orders=1):
        """
        Create a given number of randomly shuffled model orders

        Parameters
        ----------
        items : List
        num_orders : int, optional

        Returns
        -------
        orders : List[Order]
        """
        orders = list()
        for _ in range(num_orders):
            models_ = list(items)
            shuffle(models_)
            orders.append(cls([[_model] for _model in models_]))
        if len(items) > 2:
            combinations_ = [_ for _ in combinations(items, 2)]
            shuffle(combinations_)
            count = 1
            for combination in combinations_:
                remaining = list(set(items).difference(combination))
                shuffle(remaining)
                order = list()
                for model in remaining[:-1]:
                    order.append([model])
                order.append(list(combination))
                order.append(remaining[-1:])
                orders.append(cls(order))
                count += 1
                if count > num_orders:
                    break
        if len(items) > 3:
            combinations_ = [_ for _ in combinations(items, 3)]
            shuffle(combinations_)
            count = 1
            for combination in combinations_:
                remaining = list(set(items).difference(combination))
                shuffle(remaining)
                order = list()
                for model in remaining[:-1]:
                    order.append([model])
                order.append(list(combination))
                order.append(remaining[-1:])
                orders.append(cls(order))
                count += 1
                if count > num_orders:
                    break
        return orders


class Ranking(RankedOrder):
    def __init__(self, items, name=None, _rankings=None, _scores=None,
                 _weights=None, _score_dists=None, _estimators=None):
        """
        Ranking of models based on some criterion

        Parameters
        ----------
        items : List
        name : str, optional
        _rankings : List[Ranking], optional
            Ranking objects for derived rankings
        _scores : Dict[int: list], optional
            Scores to derive distribution
        _weights : List[float], optional
            KDE weights corresponding to scores
        _score_dists : Dict[int: Series], optional
            KDE-generated score distribution
        _estimators : Dict[int: KernelDensity], optional
            Kernel Density estimation objects for current state
        """
        self._items = items
        self._name = name
        ids = list()
        for item in items:
            try:
                ids.append(item.id)
            except AttributeError:
                pass
        i = 1
        try:
            i += max(ids)
        except ValueError:
            pass
        for item in items:
            try:
                item.id
            except AttributeError:
                item.id = i
                i += 1
        self.orders = list()
        self._rankings = _rankings
        self._scores = _scores
        self._weights = _weights
        self._score_dists = _score_dists
        self._estimators = _estimators
        self._bounds = None
        self._score_pdfs, self._score_cdfs = None, None
        self._pairwise_probabilities = None
        
    def __call__(self):
        """
        Need to combine the various different answers for the same thing

        Notes
        -----
        - Need to organize the list to just use sequential integers
        - Essentially not trusting anyone's scale for how much more
          something is, just the order
        - Combined ones where they are not the same in any of them will
          never be in the combined one
        - Should that just come out as an order or on a scale to show
          they are closer since at least someone said they are the same
          - Maybe this is how it should go, but how is that used later?

        Returns
        -------
        combined : Dict[int: float]
        """
        items = {_item.id: _item for _item in self.items}
        orders = [_order() for _order in self.orders]
        combined = dict()
        for key in items:
            values = [_order[key] for _order in orders]
            combined[key] = values
            # combined[key] = mean(values)
        return combined

    def __repr__(self):
        return '\n'.join(repr(order) for order in self.orders)

    def __str__(self):
        return self.name

    @staticmethod
    def _get_colors(num, cmap, reverse=False):
        cmap = cm.get_cmap(cmap)
        if cmap(0) in (cm.gray(0), cm.gray(256)):
            min_, max_ = 80 / 256., 176 / 256.
        else:
            min_, max_ = 48 / 256, 208 / 256
        return get_colors(num, cmap, reverse, min_, max_)

    @staticmethod
    def _get_weights(weights):
        weights_ = list(weights)
        while True:
            weights = list()

            def simplify(current):
                updated = list()
                for i in current:
                    if not isinstance(i, list):
                        updated.append(i)
                    else:
                        if not any(isinstance(j, list) for j in i):
                            updated.extend(normalize(i))
                        elif all(not any(isinstance(k, list) for k in j) for j
                                 in i):
                            updated.extend(flatten(i))
                        else:
                            simplify(i)
                weights.append(updated)

            simplify(weights_)
            if len(weights) == 1 and not any(
                    isinstance(i, list) for i in weights[0]):
                weights = normalize(weights[0])
                return weights
            else:
                weights_ = weights

    @classmethod
    def combine(cls, rankings, weights=None, name=None, num_samples=1000,
                score_rankings=None, score_weights=None):
        """
        Combines rankings into derived ranking using KDE

        Notes
        -----
        Weighting the rankings the same as the score rankings still
        means using KDE sample weights if they do not have the same
        number of samples (e.g. 3 fidelity metrics: 1, 2 correlation
        metrics: 1.5)

        Parameters
        ----------
        rankings : List[Ranking, size n_rankings]
        weights : List[Union[int, float], size n_rankings], optional
        name : str, optional
        num_samples : int, optional
        score_rankings : List[dict[int: Union[List[float]]]], optional
            Scores not generated from Ranking call. Iterable of
            dictionaries containing a list of samples for each item.
        score_weights : List[float, size n_scores], optional
        """
        items = list(rankings[0].items)
        score_dists = {_item.id: list() for _item in items}
        scores = dict()
        rankings_ = list(rankings)
        rankings = list()
        for ranking in rankings_:
            rankings.append(ranking())
        if weights:
            weights_ = list(weights)
        else:
            weights_ = [1 for _ in rankings]
        weights = list()
        for ranking, weight in zip(rankings, weights_):
            key = list(ranking)[0]
            weights.append([weight for _ in ranking[key]])
        if score_rankings:
            rankings.extend(score_rankings)
            weights = [weights]
            if score_weights is not None:
                if len(score_rankings) != len(score_weights):
                    raise ValueError('If provided, score rankings and weights '
                                     'must be same length')
                score_weights_ = list(score_weights)
            else:
                score_weights_ = [1 for _ in score_rankings]
            score_weights = list()
            for ranking, weight in zip(score_rankings, score_weights_):
                key = list(ranking)[0]
                score_weights.append([weight for _ in ranking[key]])
            weights.append(score_weights)
        _weights = list(weights)
        weights = cls._get_weights(weights)
        estimators = dict()
        for key in score_dists:
            scores_ = [_ranking[key] for _ranking in rankings]
            scores_ = np.array([_i for _j in scores_ for _i in _j])
            scores[key] = scores_
            params = {'bandwidth': 5 * np.logspace(-2, 1, 20)}
            grid = GridSearchCV(KernelDensity(), params, cv=len(scores_))
            grid.fit(scores_[:, None], sample_weight=weights)
            kde = grid.best_estimator_
            estimators[key] = kde

            def find_start_stop(_xs):
                """
                Find the borders of the non-zero distribution region

                Parameters
                ----------
                _xs : Array[float]

                Returns
                -------
                start, stop : int
                """
                _start, _stop = 0, -1
                _sample = np.exp(kde.score_samples(_xs[:, np.newaxis]))
                for count, value in enumerate(_sample):
                    if value > 1e-8:
                        _start = max((0, count - 1))
                        break
                for count, value in enumerate(reversed(_sample)):
                    if value > 1e-8:
                        _stop = -1 - max((0, count - 1))
                        break
                return _start, _stop

            xs = linspace(-2, 3, num_samples)
            start, stop = find_start_stop(xs)
            while True:
                if start == 0 and stop == -1:
                    break
                xs = linspace(xs[start], xs[stop], num_samples)
                start, stop = find_start_stop(xs)
            sample = np.exp(kde.score_samples(xs[:, np.newaxis]))
            score_dists[key] = Series(sample, xs)
        return cls(items, name, rankings, scores, weights, score_dists,
                   estimators)

    @property
    def bounds(self):
        if not self._bounds:
            min_, max_ = 0.0, 1.0
            for scores in self.score_dists.values():
                xs = scores.index.values
                _min, _max = min(xs), max(xs)
                if _min < min_:
                    min_ = _min
                if _max > max_:
                    max_ = _max
            self._bounds = min_, max_
        return self._bounds

    @property
    def items(self):
        return self._items

    @property
    def name(self):
        if not self._name:
            if not self._rankings:
                raise AttributeError('Name not defined')
            else:
                count = 0
                names = list()
                for ranking in self._rankings:
                    try:
                        name = ranking.name
                    except AttributeError:
                        count += 1
                    else:
                        names.append(name)
                if not count:
                    name = ', '.join(names[:-1]) + ', and {}'.format(names[-1])
                else:
                    if not names:
                        name = '{} rankings'.format(count)
                    else:
                        name = ', '.join(names)
                        name += ', and {} other'.format(count)
                        if count > 1:
                            name += 's'
                return name.title()
        else:
            return self._name.title()

    @property
    def scores(self):
        """
        Scores used to estimate densities, can be developed from expert
        rankings or, for models, updated using data

        Returns
        -------
        scores : Dict
        """
        if not self._scores:
            raise AttributeError('Order scores not yet created')
        return self._scores

    @scores.setter
    def scores(self, scores):
        """
        Setter for modified scores

        Notes
        -----
        Sets/resets bounds and distribution splines

        Parameters
        ----------
        scores : Dict
        """
        self._scores = scores
        self._bounds, self._weights = None, None
        self._score_pdfs, self._score_cdfs = dict(), dict()

    @property
    def score_cdfs(self):
        if not self._score_cdfs:
            self._score_cdfs, pdfs = dict(), self.score_pdfs
            for item in self.items:
                id_ = item.id
                pdf = pdfs[id_]
                xs = self.score_dists[id_].index.values
                x_ = xs[-1]
                ys = [pdf.integral(_x, x_) for _x in xs]
                min_, max_ = self.bounds
                if xs[0] > min_:
                    xs = np.append([min_], xs)
                    ys = np.append([0.0], ys)
                if xs[-1] < max_:
                    xs = np.append(xs, [max_])
                    ys = np.append(ys, [0.0])
                cdf = InterpolatedUnivariateSpline(xs, ys)
                self._score_cdfs[id_] = cdf
        return self._score_cdfs

    @property
    def score_dists(self):
        """
        Scores derived from expert-generated ranked orders for derived
        rankings

        Returns
        -------
        scores : Dict
        """
        if not self._score_dists:
            raise AttributeError('Order score distributions not yet created')
        return self._score_dists

    @property
    def score_pdfs(self):
        if not self._score_pdfs:
            self._score_pdfs = dict()
            for item in self.items:
                key = item.id
                scores = self.score_dists[key]
                xs = scores.index.values
                ys = scores.values
                min_, max_ = self.bounds
                if xs[0] > min_:
                    xs = np.append([min_], xs)
                    ys = np.append([0.0], ys)
                if xs[-1] < max_:
                    xs = np.append(xs, [max_])
                    ys = np.append(ys, [0.0])
                pdf = InterpolatedUnivariateSpline(xs, ys)
                self._score_pdfs[key] = pdf
        return self._score_pdfs

    @property
    def weights(self):
        """
        Weights corresponding to fidelity scores

        Returns
        -------
        weights : List
        """
        if not self._weights:
            raise AttributeError('Order weights not yet created')
        return self._weights

    @weights.setter
    def weights(self, weights):
        """
        Setter for modified weights

        Notes
        -----
        Sets/resets bounds and distribution splines

        Parameters
        ----------
        weights : List
        """
        self._weights = weights
    
    def add_order(self, order):
        """
        Add a new model order based on expert opinion

        Parameters
        ----------
        order : Order
            Dictionary of model: integer ranked order pairs. Multiple models
            can have the same ranking
        """
        self.orders.append(order)

    def add_order_decreasing(self, items):
        """
        Add model order in decreasing order as listed

        Notes
        -----
        Shortcut for add_order

        Parameters
        ----------
        items
        """
        self.orders.append(Order(list(reversed([[_] for _ in items]))))

    def add_order_fixed(self, items):
        """
        Add model order all at the same level in terms of res/abs/scope

        Notes
        -----
        Shortcut for add_order

        Parameters
        ----------
        items
        """
        self.orders.append(Order([items]))

    def add_order_increasing(self, items):
        """
        Add model order in increasing order as listed

        Notes
        -----
        Shortcut for add_order

        Parameters
        ----------
        items
        """
        self.orders.append(Order([[_] for _ in items]))

    def describe(self, key, percentiles=None):
        """
        Custom describe call for score series

        Parameters
        ----------
        key : int
            Score item key
        percentiles : List[float], optional

        Returns
        -------
        description : Series
        """
        labels = ['count', 'mean', 'mode', 'std']
        values = [len(self.score_dists[key]), self.score_mean(key),
                  self.score_mode(key), self.score_std(key)]
        if not percentiles:
            percentiles = [25, 50, 75]
        for percentile in percentiles:
            labels.append('{}%'.format(percentile))
            values.append(self.score_percentile(key, percentile))
        return Series(values, index=labels)

    def plot_ranked(self, level=1, use_names=True, cumulative=True):
        data = self.prob_level(level, use_names=use_names)
        data = Series(data, name='Probability').sort_values()
        print(data)
        fig, ax = plt.subplots()
        ax.barh(range(len(data)), data.values, tick_label=data.index,
                alpha=0.5, color='black')
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Probability (Bars)')
        xmin, xmax = ax.get_xlim()
        xmax = ceil(xmax * 10) / 10
        ax.set_xlim(xmin, xmax)
        if cumulative:
            ymin, ymax = ax.get_ylim()
            ax2 = ax.twiny()
            cumulative = data.sort_values(ascending=False).cumsum()
            ys = [_ * 100 for _ in reversed([0] + list(cumulative) + [1])]
            ax2.plot(ys, [-1] + list(range(len(data) + 1)), 'k')
            fmt = '%.0f%%'  # Format you want the ticks, e.g. '40%'
            xticks = ticker.FormatStrFormatter(fmt)
            ax2.xaxis.set_major_formatter(xticks)
            ax2.set_xlabel('Cumulative (Line)')
            ax2.vlines(100, ymin, ymax, color='black', alpha=0.5)
            ax.set_ylim(ymin, ymax)
            ax2.spines['right'].set_visible(False)

    def plot_score_bars(self, cmap=None):
        """
        Plots stacked bar charts of probabilities that each item is 1st
        through last based on scores

        Parameters
        ----------
        cmap
        """
        prob_levels = [self.prob_level(_) for _ in
                       range(1, len(self.items) + 1)]
        plt.figure()
        for i, scores in enumerate(prob_levels):
            plot_proportion(i, scores, cmap=cmap, items=self.items,
                            legend=False)
        plt.xticks(range(len(self.items)),
                   ['P({}$^{{{}}}$)'.format(i, ordinal(i, True)) for i in
                    range(1, len(self.items))] + ['P(last)'])
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    def plot_scores(self, tol=1e-5, cdf=False, cmap=None, reverse=False,
                    title=True, ax=None, scores=None, legend=True,
                    use_ids=False, _durations=False):
        """
        Plot KDE distributions and return distribution medians

        Parameters
        ----------
        tol : float, optional
        cdf : bool, optional
        cmap : optional
            Matplotlib cmap
        reverse : bool, optional
        title : str, optional
        ax : optional
            Matplotlib axis to use
        scores : dict, optional
            Model score samples. Will use .score_dists if not provided.
        legend : bool, optional
        use_ids : bool, optional
            Flag to use IDs instead of names in legend
        _durations
            Run time data, used by estimate_cost

        Returns
        -------
        medians : List
            List of tuple pairs in the form: (median, Model)
        """
        if scores is None:
            scores = self.score_dists
        if len(self.items) > 6 and not _durations:
            if reverse:
                _value = 1.0
            else:
                _value = 0.0
            colors = [cmap(_value) for _ in self.items]
        else:
            if cmap in (cm.gray, cm.gray_r):
                colors = get_colors(len(self.items), cmap, min_=80/256,
                                    max_=176/256)
            else:
                colors = get_colors(len(self.items), cmap, min_=100/256,
                                    max_=208/256)
        colors = cycle(colors)
        if not ax:
            plt.figure()
            ax = plt.gca()
        for key in ['left', 'top', 'right']:
            ax.spines[key].set_visible(False)
        if title is None:
            pass
        elif not isinstance(title, bool):
            ax.set_title(title)
        elif title:
            ax.set_title(self.name)
        all_xs, all_ys, items = list(), list(), list()
        max_ys = list()
        for model in self.items:
            sample = scores[model.id]
            sample = sample[sample > tol]
            xs, ys = [_ for _ in sample.index], [_ for _ in sample]
            median, _ = dist_median(xs,  ys)
            items.append((median, model, xs, ys))
            all_xs.extend(xs)
            all_ys.extend(ys)
            max_ys.append(max(ys))
        min_x, max_x = min(all_xs), max(all_xs)
        all_ys = Series(all_ys)
        all_ys = all_ys[all_ys < all_ys.mean() + 3 * all_ys.std()]
        max_ys = Series(max_ys)
        items.sort(key=itemgetter(0))
        model_dict = {_model.id: _model for _, _model, _, _ in items}
        xs = [_xs for _, _, _xs, _ in items]
        ys = [_ys for _, _, _, _ys in items]
        if use_ids:
            labels = [str(_model.id) for _, _model, _, _ in items]
        else:
            labels = ['{}: {}'.format(_model.id, _model) for _, _model, _, _ in
                      items]
        if len(self.items) > 6:
            if any(_ for _ in max_ys if _ > max_ys.mean() + 2 * max_ys.std()):
                height = all_ys.std() * all_ys.median() / all_ys.mean() * 1.5
            else:
                height = all_ys.var() * all_ys.median() / all_ys.mean() * 1.5
            y_ticks = list()
            for count, (label, x, y) in enumerate(zip(labels, xs, ys)):
                color = next(colors)
                y = np.array(y)
                y_adj = height * count
                y_ticks.append(y_adj)
                ax.plot(x, y + y_adj, color=color, label=label)
                if color == (1.0, 1.0, 1.0, 1.0):
                    _color = 'k'
                else:
                    _color = color
                ax.fill_between(x, [y_adj for _ in x], y + y_adj,
                                alpha=0.3, color=_color, label='_nolegend_')
            if _durations:
                ax.legend()
                ax.set_yticks(list())
                ax.set_yticklabels(list())
            elif legend:
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(labels)
            else:
                ax.set_yticks(list())
                ax.set_yticklabels(list())
            ax.set_xlim(min_x, max_x)
            if cdf:
                ax.set_title(self.name)
                ys = list()
                for model, x in zip(self.items, xs):
                    ys.append(self.score_cdfs[model.id](x))
                height = all_ys.std() * all_ys.median() / all_ys.mean() * 1.5
                y_ticks = list()
                for count, (label, x, y) in enumerate(zip(labels, xs, ys)):
                    color = next(colors)
                    y_adj = height * count
                    y_ticks.append(y_adj)
                    ax.plot(x, y + y_adj, color=color)
                    if color != (1.0, 1.0, 1.0, 1.0):
                        ax.fill_between(x, [y_adj for _ in x], y + y_adj,
                                        alpha=0.3, color=color)
                if _durations:
                    ax.legend()
                elif legend:
                    ax.set_yticks(y_ticks)
                    ax.set_yticklabels(labels)
                min_, _max = ax.get_ylim()
                max_ = ((max(ys[-2]) + max(ys[-1])) / 2 + y_ticks[-1]) * 1.1
                min_ = min_ * max_ / _max
                ax.set_ylim(min_, max_)
                ax.set_xlim(min_x, max_x)
        else:
            for _, model, xs, ys in items:
                color = next(colors)
                ax.plot(xs, ys, label='_nolegend_', color=color)
                if color != (1.0, 1.0, 1.0, 1.0):
                    ax.fill_between(xs, [0 for _ in xs], ys, alpha=0.3,
                                    color=color)
            all_scores = [_i for _j in self.scores.values() for _i in _j]
            max_count = max(all_scores.count(_i) for _i in set(all_scores))
            offset = max(all_ys) / (5 + max_count)
            handles, plotted_scores = list(), list()
            markers_ = cycle(markers)
            for key in sorted(scores):
                color = next(colors)
                marker = next(markers_)
                for count, value in enumerate(self.scores[key]):
                    if count == 0:
                        kwargs = {'label': model_dict[key].name}
                    else:
                        kwargs = dict()
                    ax.scatter(
                        (value,), (offset * plotted_scores.count(value),),
                        color=color, marker=marker, edgecolors='k', **kwargs)
                    plotted_scores.append(value)
            ax.set_yticks(list())
            if legend:
                ax.legend()
            ax.set_xlim(min_x, max_x)
            if cdf:
                ax.title(self.name)
                for _, model, xs, _ in items:
                    color = next(colors)
                    ys = self.score_cdfs[model.id](xs)
                    ax.plot(xs, ys, label=model.name, color=color)
                    if color != (1.0, 1.0, 1.0, 1.0):
                        ax.fill_between(xs, [0 for _ in xs], ys, alpha=0.3,
                                        color=color)
                if legend:
                    ax.legend()
                ax.set_xlim(min_x, max_x)
        return [(_median, _model) for _median, _model, _, _ in items]

    def prob_level(self, level=1, include=None, use_names=False):
        """
        Calculate probability item at a certain level given current
        scores

        Notes
        -----
        - Given that they are independent probabilities, can calculate
          probability that item_i is at a certain level
        - P(A or B) = P(A) + P(B) - P(A and B) = P(A) + P(B) - P(A)P(B)
          due to independence
        - Not the same as being at least at that level
        - Probability algorithm for all except lowest level based on
          inclusion-exclusion principle

        Parameters
        ----------
        level : int, optional
            Default value is one, for probability of being highest
            fidelity in the set. Can use -1 for last in set.
        include : List[int], optional

        Returns
        -------
        prob_level : Dict[int: float]
        """
        probabilities = self.ranking_probabilities()
        prob_level = list()
        if not include:
            include = set(_item.id for _item in self.items)
        if level == -1:
            level = len(include)
        item_ids = set(_item.id for _item in self.items if _item.id in include)
        if len(item_ids) - level < 0:
            raise ValueError('Invalid level')
        for id_ in item_ids:
            prob_dict = probabilities[id_]
            others = item_ids.difference({id_})
            if not len(item_ids) - level:
                ids = others
                prob_level.append(np.prod([1 - prob_dict[_id] for _id in ids]))
            else:
                cycle_ = cycle([1, -1])
                ids = [_comb for _comb in
                       combinations(others, len(item_ids) - level)]
                probability = 0.0
                count = 1
                while True:
                    sign = next(cycle_)
                    combinations_ = [set(_comb[0]).union(*_comb[1:]) for _comb
                                     in combinations(ids, count)]
                    for combination in combinations_:
                        probability_ = np.prod(
                            [prob_dict[_id] for _id in combination])
                        probability += sign * probability_
                    if len(combinations_) == 1:
                        break
                    else:
                        count += 1
                prob_level.append(probability)
        prob_level_ = list(prob_level)
        values = np.array(prob_level)
        values /= sum(values)  # normalize
        prob_level = dict()
        for id_, value in zip(item_ids, values):
            prob_level[id_] = value
        if use_names:
            prob_level_ = dict(prob_level)
            name_dict = {item.id: item.name for item in self.items}
            prob_level = dict()
            for id_ in prob_level_:
                prob_level[name_dict[id_]] = prob_level_[id_]
        return prob_level

    def ranking_probabilities(self, num_samples=1000, plot=False):
        """
        Calculate probabilities that each item scores higher than
        another so probability ranking can be calculated

        Notes
        -----
        - Based on using KDE for creation of distributions
          - Independent Gaussian distributions
          - Assumes each point generates a centered Gaussian distribution
          - KDE bandwidth is equivalent to the variance
          - P(X > Y) = P(X - Y > 0)
          - X(\mu_X, \sigma_X^2) - Y(\mu_Y, \sigma_Y^2) =
            => Z(\mu_X - \mu_Y, \sigma_X^2 + \sigma_Y^2)
          - Regenerate subtracted distribution and integrate to get
            P(X - Y > 0) then P(Y > X) = 1 - P(X - Y > 0)
        - Tested using correlation (commented section, obsolete)
          - \sigma_{X+Y}^2 = \sigma_X^2 + \sigma_Y^2 + 2\rho\sigma_X\sigma_Y
          - \rho is Pearson correlation coefficient
          - However, this is widening the spread of the distribution if
            they are more correlated, which is the opposite of the goal
            here
          - Seems to be aliasing two concepts since this distribution is
            of the fidelity score, not the distribution of the results

        Parameters
        ----------
        num_samples : int, optional
            Number of samples to use for KDE. Default value is 1000.
        plot : bool, optional
            Flag to plot results. Default value is False.

        Returns
        -------
        probabilities
        """
        if self._pairwise_probabilities:
            return self._pairwise_probabilities
        if plot:
            plt.figure()
        probabilities = {_item.id: dict() for _item in self.items}
        xs = np.linspace(-2, 2, num_samples)
        for items in combinations(self.items, 2):
            ids = sorted([_item.id for _item in items])
            estimators = [self._estimators[_id] for _id in ids]
            bandwidths = [_est.bandwidth for _est in estimators]
            bandwidth = sum(bandwidths)  # sum variance
            kde = KernelDensity(bandwidth=bandwidth)
            id0, id1 = ids
            scores = self.scores[id0] - self.scores[id1]
            kde.fit(np.array(scores)[:, None], sample_weight=self.weights)
            ys = np.exp(kde.score_samples(xs[:, np.newaxis]))
            if plot:
                plt.plot(xs, ys, label='{}, {}'.format(*ids))
                plt.fill_between(xs, [0 for _ in xs], ys, alpha=0.3)
            spline = InterpolatedUnivariateSpline(xs, ys)
            prob = spline.integral(0, 2)
            probabilities[id0][id1] = prob
            probabilities[id1][id0] = 1 - prob
        if plot:
            plt.legend()
            _, y_max = plt.ylim()
            plt.vlines(0, 0, y_max)
            plt.yticks(list())
        self._pairwise_probabilities = probabilities
        return probabilities

    def score_mean(self, key):
        xs = self.score_dists[key].index.values
        ys = self.score_dists[key].values
        spline = InterpolatedUnivariateSpline(xs, xs * ys)
        return spline.integral(xs[0], xs[-1])

    def score_median(self, key):
        return self.score_percentile(key, 50)

    def score_mode(self, key):
        return self.score_dists[key].idxmax()

    def score_percentile(self, key, percent):
        """
        Find quantile of distribution curve

        Notes
        -----
        Percentages are measured from x = 1 down

        Parameters
        ----------
        key : int
        percent : float
            Percentage value from 0-100%

        Returns
        -------
        quantile : float
        """
        xs = self.score_dists[key].index.values
        spline = self.score_pdfs[key]
        total = spline.integral(xs[0], xs[-1])

        def fun(x):
            area = total - spline.integral(xs[0], x)
            return area / total - percent / 100

        return brentq(fun, xs[0], xs[-1])

    def score_std(self, key):
        return sqrt(self.score_var(key))

    def score_var(self, key):
        return self._estimators[key].bandwidth
        # mu = self.score_mean(key)
        # xs = self.score_dists[key].index.values
        # ys = self.score_dists[key].values
        # spline = InterpolatedUnivariateSpline(xs, (xs - mu) ** 2 * ys)
        # return spline.integral(xs[0], xs[-1])


class ModelRanking(Ranking):
    permutation_limit = 9, 6  # model limit, permutation limit

    def __init__(self, models, name=None, _rankings=None, _scores=None,
                 _weights=None, _score_dists=None, _estimators=None):
        """
        Ranking of models based on some criterion

        Parameters
        ----------
        models : List[Model]
        name : str, optional
        _rankings : List[Ranking], optional
            Ranking objects for derived rankings
        _scores : Dict[int: list], optional
            Scores to derive distribution
        _weights : List[float], optional
            KDE weights corresponding to scores
        _score_dists : Dict[int: Series], optional
            KDE-generated score distribution
        _estimators : Dict[int: KernelDensity], optional
            Kernel Density estimation objects for current state
        """
        super().__init__(models, name, _rankings, _scores, _weights,
                         _score_dists, _estimators)

    @property
    def models(self):
        return self.items

    def correlation_scoring(self, input_columns, output_columns,
                            remove_duplicates=True):
        """
        Calculate the coefficient of determination and root mean squared
        error for each model WRT all of the others and sum the totals

        Notes
        -----
        - This supposes independent of any other ranking that each model
          is the truth model and describes the relative association to
          the other models
        - Going to use median based on literature, though based on a
          small amount of testing there doesn't seem to be much
          difference

        Parameters
        ----------
        input_columns : List[str]
        output_columns : List[str]
            Column names for output values
        remove_duplicates : bool, optional

        Returns
        -------
        scores : dict[int: tuple[float]]
            Model ID associated with an R^2 and RMSE total
        """
        r2s = {_model.id: dict() for _model in self.items}
        rmses = {_model.id: dict() for _model in self.items}
        for model in self.items:
            for model_ in self.items:
                if model == model_:
                    continue
                inputs, (outputs, outputs_) = model.get_shared(
                    model_, input_columns, output_columns, remove_duplicates)
                r2s[model.id][model_.id] = r2_score(outputs, outputs_)
                r2s[model_.id][model.id] = r2_score(outputs_, outputs)
                rmses[model.id][model_.id] = mean_squared_error(
                    outputs, outputs_)
                rmses[model_.id][model.id] = mean_squared_error(
                    outputs_, outputs)
        correlations = r2s, rmses
        r2s_ = np.zeros((len(r2s), len(r2s)))
        rmses_ = np.zeros((len(rmses), len(rmses)))
        for i in r2s:
            for j in r2s[i]:
                r2s_[i - 1, j - 1] = r2s[i][j]
                rmses_[i - 1, j - 1] = rmses[i][j]
        """
        Normalized by pred, unsorted
        ----------------------------
        - Find absolute value row sums of transposed values
        - Add row sums to transposed array
        - Divide by current absolute value row sums minus original sums
        - Un-transpose
        - Subtract out the diagonal
        - Convert from array to dictionary, ignoring diagonal
        """
        r2_row_sum = np.sum(abs(r2s_.T), 1)[:, None]
        r2_scores_ = r2s_.T + r2_row_sum
        r2_scores_ /= np.sum(r2_scores_, 1)[:, None] - r2_row_sum
        r2_scores_ = r2_scores_.T
        # r2_scores_ -= np.diag(np.diag(r2_scores_))  # unecessary, ignored
        r2_scores = {_id: list() for _id in r2s}
        for i in r2s:
            for j, value in enumerate(r2_scores_[i - 1, :]):
                if i - 1 == j:
                    continue
                r2_scores[i].append(value)
        rmse_row_sum = np.sum(abs(rmses_.T), 1)[:, None]
        rmse_scores_ = -rmses_.T + rmse_row_sum
        rmse_scores_ /= np.sum(rmse_scores_, 1)[:, None] - rmse_row_sum
        rmse_scores_ = rmse_scores_.T
        # rmse_scores_ -= np.diag(np.diag(rmse_scores_))  # unecessary, ignored
        rmse_scores = {_id: list() for _id in rmses}
        for i in rmses:
            for j, value in enumerate(rmse_scores_[i - 1, :]):
                if i - 1 == j:
                    continue
                rmse_scores[i].append(value)
        scores = r2_scores, rmse_scores
        return correlations, scores

    def _efficiency_scoring(
            self, input_columns, time_columns=None, remove_outlier=True,
            plot=False, cmap=None, use_kde=False):
        costs = self.estimate_cost(
            time_columns, remove_outlier, plot, cmap, use_kde)
        crs = self.estimate_cost_ratios(
            input_columns, time_columns, remove_outlier, plot, cmap, use_kde)
        costs_ = dict()
        for i in costs.index:
            costs_[i] = costs.Cost.loc[i]
        pivot = 16.0 / 19.0
        # pivot = 1
        c1 = 1.75
        o1 = 0
        c2 = c1
        o2 = max((0, 1 - (1 + pivot) / c2))

        def fun(order):
            eff = sum(costs_[_i] for _i in order)
            i = order[0]
            for j in order[1:]:
                cr = crs[i, j]
                if cr < pivot:
                    eff *= c1 / (1 + 1 / cr) + o1
                else:  # >= pivot
                    eff *= (1 + cr) / c2 + o2
                i = j
            return eff

        return fun

    def efficiency_scoring(
            self, input_columns, time_columns, remove_outlier=True, plot=False,
            cmap=None, threshold=80, use_kde=False, freq_only=False,
            dark_background=False):
        score = self._efficiency_scoring(
            input_columns, time_columns, remove_outlier, plot, cmap, use_kde)
        ids = [_model.id for _model in self.items]
        scores = list()
        stop = self._get_stop(len(ids))
        for i in range(1, stop):
            for order in permutations(ids, i):
                eff = score(order)
                scores.append((', '.join(str(_) for _ in order), eff))
        scores = DataFrame(scores, columns=['order', 'score'])
        scores = scores.sort_values('score', ascending=True)
        scores.index = range(1, len(scores) + 1)
        scores['cumpercentage'] = (
                scores['score'].cumsum() / scores['score'].sum() * 100)
        if plot:
            plt.figure()
            if not freq_only:
                if dark_background:
                    cmap_ = cm.gray_r
                else:
                    cmap_ = cm.gray
                color = cmap_(0.0)
                plt.subplot(1, 5, (1, 3))
                if len(self.items) < 6:
                    plt.bar(scores.index, scores.score, color=color)
                else:
                    plt.plot(scores.index, scores.score, color=color)
                    if color == (1.0, 1.0, 1.0, 1.0):
                        _color = 'k'
                    else:
                        _color = color
                    plt.fill_between(
                        scores.index, [0 for _ in scores.index], scores.score,
                        color=cmap_(0), alpha=0.4)
                    plt.yscale('log')
                ax = plt.gca()
                ax2 = ax.twinx()
                ax2.plot(scores.index, scores.cumpercentage, color=cmap_(0))
                ax2.yaxis.set_major_formatter(PercentFormatter())
                ax.set_ylabel('Order score')
                plt.xticks(list())
                plt.subplot(1, 5, (4, 5))
            frequency = {_id: 0 for _id in sorted(ids)}
            for order in scores[scores.cumpercentage < threshold].order:
                for id_ in [int(_) for _ in order.split(', ')]:
                    frequency[id_] += 1
            bottom, total = 0, sum(frequency.values())
            colors = iter(get_colors(len(self.items), cmap, max_=200/256))
            for id_ in sorted(ids):
                freq = frequency[id_]
                bar = plt.bar(0, freq, bottom=bottom, color=next(colors))
                ax = plt.gca()
                for rect in bar:
                    ax.text(rect.get_x() + rect.get_width() / 2.0,
                            rect.get_y() + rect.get_height() / 2.0,
                            '{}: {:3f}%'.format(id_, freq / total),
                            ha='center', va='center')
                bottom += freq
            plt.yticks(list())
            plt.xticks([0], ['{}% Freq'.format(threshold)])
        return scores

    def estimate_cost(self, time_columns=None, remove_outlier=True,
                      plot=False, cmap=None, use_kde=False, use_ids=False,
                      tight=False, save=False):
        if time_columns is not None:
            for model in self.items:
                model.data['_t'] = sum(
                    model.data[_col] for _col in time_columns)
        ids = [_model.id for _model in self.items]
        costs, stds = list(), list()
        if plot:  # distributions
            scores = dict()
        for model in self.items:
            ts = model.data._t
            if remove_outlier:
                # print(repr(model))
                # print('Before: {}'.format(len(ts)))
                ts = remove_outliers(ts)
                # print('After: {}'.format(len(ts)))
            if use_kde:
                kde = get_kde(ts)
                median, _ = get_kde_percentile(ts, kde, 50)
                std = kde.bandwidth ** 0.5
            else:
                median = ts.median()
                std = ts.std()
            costs.append((str(model), median))
            stds.append(std)
            if plot:  # distributions, TODO: when only one sample
                if not use_kde:
                    kde = get_kde(ts)
                var = 5 * kde.bandwidth ** 0.5
                min_, max_ = min(ts) - var, max(ts) + var
                xs = np.linspace(min_, max_, 1000)
                ys = np.exp(kde.score_samples(xs[:, None]))
                scores[model.id] = Series(ys, xs)
        if plot:
            self.plot_scores(cmap=cmap, title=False, scores=scores,
                             _durations=True, use_ids=use_ids)
            ax = plt.gca()
            ax.set_xlabel('Cost (s)')
            if save:
                plt.savefig('{}-cost-dists-{}{}'.format(
                    self.name.lower(), len(self.items), save))
        if plot:  # bar chart
            _costs = [(_str, _median, _std) for (_str, _median), _std in
                      zip(costs, stds)]
            _costs = DataFrame(_costs, index=ids,
                               columns=['Model', 'Cost', 'Err'])
            colors = self._get_colors(len(self.items), cmap)
            plt.figure()
            _costs = _costs.sort_values('Cost')
            plt.bar(_costs.index, _costs['Cost'], color=colors)
            plt.xlabel('Model ID')
            plt.ylabel('Estimated Cost (s)')
            plt.xticks(range(1, len(self.items) + 1),
                       range(1, len(self.items) + 1))
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if tight:
                plt.tight_layout()
            if save:
                plt.savefig('{}-costs-{}{}'.format(
                    self.name.lower(), len(self.items), save))
        costs = DataFrame(costs, index=ids, columns=['Model', 'Cost'])
        return costs

    def estimate_cost_ratios(self, input_columns, time_columns=None,
                             remove_outlier=True, plot=False, cmap=None,
                             use_kde=False):
        # TODO: find median of duplicates?
        if time_columns is None:
            time_columns = ['_t']
            if any('_t' not in _model.data.columns for _model in self.items):
                raise ValueError('Times not yet compiled, must define columns')
        else:
            for model in self.items:
                model.data['_t'] = sum(
                    model.data[_col] for _col in time_columns)
        crs = dict()
        for model, model_ in combinations(self.items, 2):
            inputs, (outputs, outputs_) = model.get_shared(
                model_, input_columns, time_columns, True)
            cr, cr_ = outputs / outputs_, outputs_ / outputs
            cr, cr_ = cr, cr_
            if remove_outlier:
                cr = remove_outliers(Series(cr))
                cr_ = remove_outliers(Series(cr_))
            if use_kde:
                kde, kde_ = get_kde(cr), get_kde(cr_)
                median, _ = get_kde_percentile(cr, kde, 50)
                median_, _ = get_kde_percentile(cr_, kde_, 50)
            else:
                median, median_ = np.median(cr), np.median(cr_)
            crs[model_.id, model.id] = median
            crs[model.id, model_.id] = median_
        # crs_ = np.ones((len(self.items), len(self.items)))  # array
        # for i in range(len(self.items)):
        #     for j in range(len(self.items)):
        #         if i == j:
        #             continue
        #         crs_[i, j] = crs[i + 1, j + 1]
        if plot:  # bar chart
            crs_ = list()
            for i in range(len(self.items)):
                for j in range(len(self.items)):
                    if i >= j:
                        continue
                    min_, max_ = sorted((crs[i + 1, j + 1],
                                         1 / crs[j + 1, i + 1]))
                    crs_.append(('{},{}'.format(i + 1, j + 1), min_, max_))
            crs_ = DataFrame(crs_, columns=('Pair', 'C_r1', 'C_r2'))
            crs_['Mean C_r'] = crs_[['C_r1', 'C_r2']].mean(axis=1)
            crs_['e1'] = crs_['Mean C_r'] - crs_['C_r1']
            crs_['e2'] = crs_['C_r2'] - crs_['Mean C_r']
            colors = self._get_colors(len(crs_), cmap)
            plt.figure()
            plt.bar(crs_.index, crs_['Mean C_r'], color=colors,
                    yerr=crs_[['e1', 'e2']].values.T)
            plt.xlabel('Model ID')
            plt.ylabel('Mean $C_r$')
            plt.xticks(list())
            plt.xticks(range(len(crs_)),
                       [_.replace(',', '\n') for _ in crs_['Pair']])
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        return crs

    def get_duplicate_models(
            self, input_columns, output_columns, r2_tol=0.95, rmse_tol=1e-2,
            remove_duplicates=True, kernel=None, plot=False):
        """
        Compare the correlation values for the model combinations to
        determine if any can be removed due to duplicate results

        Notes
        -----
        Model probabilities or required effort should be used to
        determine which model from a pair should be removed

        Parameters
        ----------
        input_columns : List[str]
        output_columns : List[str]
        r2_tol : float, optional
        rmse_tol : float, optional
        remove_duplicates : bool, optional
        kernel : optional
        plot : bool, optional

        Returns
        -------
        duplicates : List[tuple[model]]
        """
        (r2s, rmses), _ = self.correlation_scoring(
            input_columns, output_columns, remove_duplicates)
        duplicate_set = set()
        for key in sorted(r2s):
            for other in sorted(r2s[key]):
                if r2s[key][other] > r2_tol and rmses[key][other] < rmse_tol:
                    print(key, other, r2s[key][other], rmses[key][other])
                    duplicate_set.add(tuple(sorted([key, other])))
        duplicates = list()
        for couple_ in duplicate_set:
            couple = list()
            for id_ in couple_:
                for model in self.items:
                    if model.id == id_:
                        couple.append(model)
            duplicates.append(tuple(couple))
        if plot:
            _rmses = list()
            for key in rmses:
                _rmses.extend(rmses[key].values())
            quantile = Series(_rmses).quantile(0.25)
            _, axarr = plt.subplots(1, 2)
            axarr[0].hlines(r2_tol, -1, 1)
            axarr[0].set_xlim(-1, 1)
            axarr[0].set_ylim(0, 1)
            axarr[0].set_xticks(list())
            axarr[0].set_xlabel('$R^2$')
            axarr[1].hlines(rmse_tol, -1, 1)
            axarr[1].set_xlim(-1, 1)
            axarr[1].set_ylim(0, quantile)
            axarr[1].set_xticks(list())
            axarr[1].set_xlabel('RMSE')
            # axarr[1].set_yscale('log')
            offset, jitter, sign = 0, 1 / len(self.items), +1
            r2_set, rmse_set = set(), set()
            couples = set()
            for key in sorted(r2s):
                for other in sorted(r2s[key]):
                    couple = tuple(sorted([key, other]))
                    if couple in couples:
                        continue
                    else:
                        couples.add(couple)
                    label = ', '.join(str(_) for _ in couple)
                    r2 = r2s[key][other]
                    if r2 > 0:
                        if r2 not in r2_set:
                            x = 0
                            r2_set.add(r2)
                        else:
                            offset += jitter
                            x = offset * sign
                            sign *= -1
                        axarr[0].scatter((x,), (r2,), label=label)
                    rmse = rmses[key][other]
                    if rmse <= quantile:
                        if rmse not in rmse_set:
                            x = 0
                            rmse_set.add(rmse)
                        else:
                            offset += jitter
                            x = offset * sign
                            sign *= -1
                        axarr[1].scatter((x,), (rmse,), label=label)
            axarr[0].legend()
            axarr[1].legend()
        return duplicates

    def get_model_gaussian_regressors(
            self, input_columns, output_columns, kernel=None, num_test=3,
            _io=None):
        """
        Create the single model Gaussian process regressions and two-
        model delta regressions for Co-Kriging regression building

        Parameters
        ----------
        input_columns : iterable[str]
            Column names for input values
        output_columns : iterable[str]
            Column names for output values
        kernel : kernel object, optional
            The kernel specifying the covariance function of the GP. If
            None is passed, the kernel "1.0 * RBF(1.0)" is used as
            default. Note that the kernel's hyperparameters are
            optimized during fitting. Default value is ConstantKernel * RBF.
        num_test : int, optional
            Number of length scale values in logspace to check for global
            max log marginal likelihood. Default value is 3.
        _io : tuple[dict[int: array[float]]], optional

        Returns
        -------
        regressors : dict[int, tuple[int]:
                           GaussianProcessRegressor, DeltaGPR]
        """
        regressors = dict()
        if _io:
            inputs, outputs = _io
        else:
            inputs, outputs = self.map_io(input_columns, output_columns)
        for model in self.items:
            io = inputs[model.id], outputs[model.id]
            regressors[model.id] = model.get_gpr(
                input_columns, output_columns, kernel, num_test, io)
        return regressors

    def _get_stop(self, size):
        model_lim, perm_lim = self.permutation_limit
        if size <= model_lim:
            stop = size + 1
        else:
            warn('Permutation limit exceeded, orders truncated')
            stop = max((perm_lim, model_lim - (size - model_lim)))
        return stop

    def map_io(self, input_columns, output_columns, remove_duplicates=True):
        """
        Map the column names of inputs and outputs to dictionaries of
        values

        Notes
        -----
        Output array is 1-D if only one response
        Input array is always 2-D
        This is for working with scikit-learn

        Parameters
        ----------
        input_columns : iterable[str]
            Column names for input values
        output_columns : iterable[str]
            Column names for output values
        remove_duplicates : bool, optional

        Returns
        -------
        inputs, outputs : dict[int: array[float]]
        """
        inputs, outputs = dict(), dict()
        for model in self.items:
            try:
                inputs_, outputs_ = model.map_io(
                    input_columns, output_columns, remove_duplicates)
            except KeyError:
                warn('IO KeyError for model: {}'.format(model))
            else:
                inputs[model.id] = inputs_
                outputs[model.id] = outputs_
        return inputs, outputs

    def multiattribute_scoring(
            self, input_columns, time_columns=None, remove_outlier=True,
            use_kde=False, plot=False, cmap=None, plot_single=False):
        """
        TODO: add capability to use more than one fidelity estimate

        Parameters
        ----------
        input_columns
        time_columns
        remove_outlier
        use_kde
        plot
        cmap
        plot_single

        Returns
        -------
        dominant
        """
        fid_off, stop = 0.01, 1e-6
        score_fid = self._multifidelity_scoring()
        score_eff = self._efficiency_scoring(
            input_columns, time_columns, remove_outlier, False, cmap, use_kde)
        ids, order_scores = [_model.id for _model in self.items], list()
        single = [(_id, score_fid((_id,)), score_eff((_id,))) for _id in ids]
        single = DataFrame(single, columns=('order', 'fidelity', 'efficiency'))
        single['_fidelity'] = 1 / single.fidelity
        single['_efficiency'] = 1 / single.efficiency
        indices = pareto_front(single[['_fidelity', 'efficiency']].values)
        single_dom = single.loc[indices]
        # print(single_dom)
        eff_off = min((log(single.efficiency.max()),
                       single.efficiency.max() * fid_off))
        if plot and plot_single:
            color = cmap(0.0)
            plt.figure()
            texts = list()
            for _, order, fid, eff, _, _ in single.itertuples():
                plt.scatter((eff,), (fid,), color=color)
                texts.append(plt.text(eff, fid, order))
            plt.plot(single_dom.efficiency,
                     single_dom.fidelity, color=color, label='Single-Model')
            plt.fill_between(
                single_dom.efficiency,
                [min(single_dom.fidelity) for _ in indices],
                single_dom.fidelity, color=color, alpha=0.3)
            adjust_text(texts)
        columns = (
            'order', 'fidelity', 'efficiency', '_fidelity', '_efficiency')
        dominant = None

        def _update(_current, _new):
            if _current is None:
                _current = np.array(_new)
                _start = len(_current)
            else:
                _start = len(_current)
                _current = np.concatenate((_current, np.array(_new)))
            _check = _current[:, 2:4].astype(float)
            indices = pareto_front(_check)
            _updated = _current[indices]
            _diff = len(indices) - _start
            print(_start, _diff)
            return _updated

        count = 0
        for i in range(2, len(self.items) + 1):
            t0 = time()
            j, new_scores = 0, list()
            num = num_permutations(len(self.items), stop=i - 1)
            for k, order in enumerate(permutations(ids, i)):
                key = ', '.join(str(_) for _ in order)
                fidelity = score_fid(order)
                efficiency = score_eff(order)
                new_scores.append((key, fidelity, efficiency,
                                   1 / fidelity, 1 / efficiency))
                count += 1
                j += 1
                if j > 1e6:
                    dominant = _update(dominant, new_scores)
                    if len(dominant) / (num + k) <= stop:
                        break
                    j, new_scores = 0, list()
            dominant = _update(dominant, new_scores)
            num_ = num + k
            ratio = len(dominant) / num_
            print('1:{:,} in {:,} permutations'.format(int(1 / ratio), num_))
            if ratio <= stop:
                warn('Permutation limit exceeded after {:,} permutations. '
                     'Max length: {}'.format(num_, i))
                break
            else:
                print('Size {} model combinations took {} seconds'.format(
                    i, time() - t0))
        print('{} permutations evaluated'.format(count))
        efficiency = dominant[:, 2].astype(float)
        fidelity = dominant[:, 1].astype(float)
        if plot:
            color = cmap(50 / 256)
            texts = list()
            for order, fid, eff in zip(*dominant[:, :3].T):
                eff, fid = float(eff), float(fid)
                plt.scatter(float(eff), float(fid), color=color)
                if fid > 0.2:  # can change tol
                    texts.append(plt.text(eff, fid, order))
            plt.plot(efficiency, fidelity, color=color, linestyle='--',
                     label='Multifidelity')
            if len(efficiency) > 1:
                plt.fill_between(
                    efficiency, [min(fidelity) for _ in dominant],
                    fidelity, color=color, alpha=0.3)
            adjust_text(texts)
            ax = plt.gca()
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            plt.legend()
            plt.xlabel('Cost/Efficiency Score (s)')
            plt.ylabel('Fidelity Score')
        dominant = DataFrame(dominant, columns=columns)
        for column in columns[1:]:
            dominant[column] = dominant[column].astype(float)
        return dominant

    def _multifidelity_scoring(self):
        """

        Returns
        -------
        fun
        """
        ids = [_model.id for _model in self.items]
        """ Option 1: original, model probabilities directly """
        # prob_first = dict()
        # for i in range(2, len(ids) + 1):
        #     for combination in combinations(ids, i):
        #         key = tuple(sorted(combination))
        #         prob_first[key] = self.prob_level(include=combination)
        """ Option 2: all prob levels normalized to local max """
        # probs = self.prob_level()
        # max_ = max(probs.values())
        # for key in probs:
        #     probs[key] /= max_
        # prob_first = {tuple(sorted(ids)): probs}
        # for i in range(2, len(ids)):
        #     for combination in combinations(ids, i):
        #         key = tuple(sorted(combination))
        #         probs_ = self.prob_level(include=combination)
        #         max_ = max(probs_.values())
        #         for key_ in probs_:
        #             probs_[key_] /= max_
        #         prob_first[key] = probs_
        """ Option 3: normalized but scaled by overall score """
        # probs = self.prob_level()
        # max_ = max(probs.values())
        # for key in probs:
        #     probs[key] /= max_
        # prob_first = {tuple(sorted(ids)): probs}
        # for i in range(2, len(ids)):
        #     for combination in combinations(ids, i):
        #         key = tuple(sorted(combination))
        #         probs_ = self.prob_level(include=combination)
        #         id_, max_ = None, 0
        #         for key_ in probs_:
        #             value = probs_[key_]
        #             if value > max_:
        #                 id_, max_ = key_, value
        #         for key_ in probs_:
        #             probs_[key_] /= max_
        #             probs_[key_] *= probs[id_]
        #         prob_first[key] = probs_
        # """
        # Option 4
        # --------
        # Normalized but scaled by average between overall score and 1.
        # This fixes the issue for picking the top two models since
        # selecting them in the appropriate order is higher scoring than
        # vice versa, and the combination of the top two models
        # appropriately scores higher than the fidelity of the 2nd place
        # model.
        # Is this assuming that you picked the highest fidelity model 1st?
        # """
        # probs = self.prob_level()
        # max_ = max(probs.values())
        # for key in probs:
        #     probs[key] /= max_
        # prob_first = {tuple(sorted(ids)): probs}
        # for i in range(1, len(ids)):
        #     for combination in combinations(ids, i):
        #         key = tuple(sorted(combination))
        #         probs_ = self.prob_level(include=combination)
        #         id_, max_ = None, 0
        #         for key_ in probs_:
        #             value = probs_[key_]
        #             if value > max_:
        #                 id_, max_ = key_, value
        #         for key_ in probs_:
        #             probs_[key_] /= max_
        #             probs_[key_] *= (1 + probs[id_]) / 2
        #         prob_first[key] = probs_
        """
        Option 5: adjusted Option 4
        ---------------------------
        Normalized, scaled by average to fix issue with selecting top 2
        options, but average can be weighted to allow for variation
        """
        w = 1.0
        probs = self.prob_level()
        max_ = max(probs.values())
        for key in probs:
            probs[key] /= max_
        prob_first = {tuple(sorted(ids)): probs}
        # print('Single model scores: ' + str(prob_first.values()))
        for i in range(1, len(ids)):
            for combination in combinations(ids, i):
                key = tuple(sorted(combination))
                probs_ = self.prob_level(include=combination)
                id_, max_ = None, 0
                for key_ in probs_:
                    value = probs_[key_]
                    if value > max_:
                        id_, max_ = key_, value
                for key_ in probs_:
                    probs_[key_] /= max_
                    probs_[key_] *= (1 + w * probs[id_]) / (1 + w)
                prob_first[key] = probs_

        def fun(order):
            available = set(ids)
            score = 1
            for id_ in order:
                key = tuple(sorted(available))
                try:
                    score *= prob_first[key][id_]
                except KeyError:  # only one model available
                    pass  # *= 1.0
                else:
                    available.remove(id_)
            return score

        return fun

    def multifidelity_scoring(
            self, plot=False, threshold=80, cmap=None, freq_only=False,
            dark_background=False):
        """

        Parameters
        ----------
        plot
        threshold
        cmap
        freq_only
        dark_background

        Returns
        -------
        scores
        """
        start = 1
        get_score = self._multifidelity_scoring()
        ids = [_model.id for _model in self.items]
        scores = list()
        stop = self._get_stop(len(self.items))
        for i in range(start, stop):
            for order in permutations(ids, i):
                score = get_score(order)
                scores.append((', '.join(str(_) for _ in order), score))
        scores = DataFrame(scores, columns=['order', 'score'])
        scores = scores.sort_values('score', ascending=False)
        scores.index = range(1, len(scores) + 1)
        scores['cumpercentage'] = (
                scores['score'].cumsum() / scores['score'].sum() * 100)
        if plot:
            plt.figure()
            if not freq_only:
                if dark_background:
                    cmap_ = cm.gray_r
                else:
                    cmap_ = cm.gray
                color = cmap_(0.0)
                plt.subplot(1, 5, (1, 3))
                if len(self.items) < 6:
                    plt.bar(scores.index, scores.score, color=color)
                else:
                    plt.plot(scores.index, scores.score, color=color)
                    if color == (1.0, 1.0, 1.0, 1.0):
                        _color = 'k'
                    else:
                        _color = color
                    plt.fill_between(
                        scores.index, [0 for _ in scores.index], scores.score,
                        color=_color, alpha=0.4)
                    plt.yscale('log')
                ax = plt.gca()
                ax2 = ax.twinx()
                ax2.plot(scores.index, scores.cumpercentage, color=cmap_(0))
                ax2.yaxis.set_major_formatter(PercentFormatter())
                ax.set_ylabel('Order score')
                plt.xticks(list())
                plt.subplot(1, 5, (4, 5))
            frequency = {_id: 0 for _id in sorted(ids)}
            for order in scores[scores.cumpercentage < threshold].order:
                for id_ in [int(_) for _ in order.split(', ')]:
                    frequency[id_] += 1
            bottom, total = 0, sum(frequency.values())
            if cmap in (cm.gray, cm.gray_r):
                colors = get_colors(len(self.items), cmap, min_=80/256,
                                    max_=176/256)
            else:
                colors = get_colors(len(self.items), cmap, min_=48/256,
                                    max_=208/256)
            colors = iter(colors)
            for id_ in sorted(ids):
                freq = frequency[id_]
                bar = plt.bar(0, freq, bottom=bottom, color=next(colors))
                ax = plt.gca()
                for rect in bar:
                    if freq / total > 0.03:
                        ax.text(rect.get_x() + rect.get_width() / 2.0,
                                rect.get_y() + rect.get_height() / 2.0,
                                '{}: {:.3f}%'.format(id_, freq / total),
                                ha='center', va='center')
                bottom += freq
            plt.yticks(list())
            plt.xticks([0.0], ['{}% Freq'.format(threshold)])
        return scores


