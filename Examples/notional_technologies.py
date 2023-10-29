import matplotlib.pyplot as plt
from pandas import DataFrame
from FAMS.utils import pareto_front
from FAMS.model_rankings import Technology, Ranking

num_technologies = 5
num_metrics = 2

technologies = [Technology(id_=i) for i in range(num_technologies)]

metric1s = list()

metric1 = Ranking(technologies, 'Metric 1')
metric1.add_order_increasing(technologies)
metric1s.append(metric1)
metric1 = Ranking(technologies, 'Metric 1')
metric1.add_order_increasing(technologies)
metric1s.append(metric1)
metric1 = Ranking(technologies, 'Metric 1')
metric1.add_order_fixed(technologies)
metric1s.append(metric1)

metric1 = Ranking.combine(metric1s, name='Metric 1')

# metric1.plot_scores()
# plt.show()
# metric1.plot_score_bars()
# plt.show()

metric2s = list()

metric2 = Ranking(technologies, 'Metric 2')
metric2.add_order_decreasing(technologies)
metric2s.append(metric2)
metric2 = Ranking(technologies, 'Metric 2')
metric2.add_order_fixed(technologies)
metric2s.append(metric2)

metric2 = Ranking.combine(metric2s, name='Metric 2')

# metric2.plot_scores()
# plt.show()
# metric2.plot_score_bars()
# plt.show()

data = DataFrame({metric.name: metric.prob_level()
                  for metric in (metric1, metric2)})
print(data)
print(pareto_front(data.to_numpy()))
data.plot.scatter('Metric 1', 'Metric 2')
plt.show()

"""
TODO
----
- Horizontal bar chart for metric
- Add pareto front step to scatter
"""
