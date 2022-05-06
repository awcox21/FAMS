import pandas as pd

from notional_expert_ranking import *
import numpy as np


def exponential_time_variation():
    return np.logspace(0, 3, len(models))


def linear_time_variation():
    return np.linspace(1, 100, len(models))


def small_time_variation():
    return np.linspace(100, 101, len(models))


notional_response = list()
xs = np.linspace(0, 100, 100)
y_list = list()
for _ in models[:-1]:
    while True:
        a, b, c, d = (random.random() for _ in range(4))
        if b:  # divides by b in random line equation
            break
    y_list.append(10 * a * xs + 25 * b * np.sin(xs / 10 / b) +
                  10 * c * xs ** (2 * c) + 100 * d)  # random curved line
y_list.append(y_list[-1] + 25)  # guarantee last two have similar shape

if plot:  # plot notional output data
    plt.figure()
    for ys in y_list:
        plt.plot(xs, ys)
    plt.title('Notional model set response')

first = True
input_columns = 'Input',
output_columns = 'Output',
time_columns = 'Time',
for model_times in (exponential_time_variation(),
                    linear_time_variation(),
                    small_time_variation()):
    for model, ys, time in zip(models, y_list, model_times):
        durations = [random.normalvariate(time, time / 10) for _ in ys]
        model.data = pd.DataFrame({'Time': durations,
                                   'Input': xs,
                                   'Output': ys})
    # multifidelity scoring from expert-opinion
    multi_expert_scores = combined.multiattribute_scoring(
        input_columns, time_columns, plot=plot, cmap=mono_cmap, plot_single=True)
    # correlation scoring
    correlations, correlation_scores = combined.correlation_scoring(
        input_columns, output_columns)
    r2s, rmses = correlations
    r2_scores, rmse_scores = correlation_scores
    # multifidelity correlation scoring
    adjusted = Ranking.combine(notional_rankings, name='correlation adjusted',
                               score_rankings=[r2_scores, rmse_scores])
    adjusted.multiattribute_scoring(
        input_columns, time_columns, plot=plot, cmap=mono_cmap, plot_single=True)

if __name__ == '__main__':
    plt.show()
