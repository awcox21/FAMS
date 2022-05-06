"""
Create initial fidelity ranking based on expert opinion orders

Contains multiple functions to depict different scenarios of expert
model rankings in terms of resolution, abstraction, and scope to be
combined
"""
from notional_models import *
from FAMS.model_rankings import Ranking


def no_scope():
    """
    Only resolution and abstraction, both purely increasing

    Notes
    -----
    Orders should typically be added using Order from model_rankings
    and Ranking.add_order, shortcut methods are used here for these
    simplified cases

    Returns
    -------
    individual_ranking
    """
    resolution = Ranking(models, 'Resolution')
    resolution.add_order_increasing(models)
    abstraction = Ranking(models, 'Abstraction')
    abstraction.add_order_increasing(models)
    return resolution, abstraction


def fixed_scope():
    """
    Resolution, abstraction, and scope, with purely increasing
    resolution and abstraction and fixed scope

    Notes
    -----
    Orders should typically be added using Order from model_rankings
    and Ranking.add_order, shortcut methods are used here for these
    simplified cases

    Returns
    -------
    individual_ranking
    """
    resolution = Ranking(models, 'Resolution')
    resolution.add_order_increasing(models)
    abstraction = Ranking(models, 'Abstraction')
    abstraction.add_order_increasing(models)
    scope = Ranking(models, 'Scope')
    scope.add_order_fixed(models)
    return resolution, abstraction, scope


def decreasing_scope():
    """
    Resolution, abstraction, and scope, with purely increasing
    resolution and abstraction and purely decreasing scope, representing
    a realistic scenario where scope has to be decreased as resolution
    and abstraction are improved

    Notes
    -----
    Orders should typically be added using Order from model_rankings
    and Ranking.add_order, shortcut methods are used here for these
    simplified cases

    Returns
    -------
    individual_ranking
    """
    resolution = Ranking(models, 'Resolution')
    resolution.add_order_increasing(models)
    abstraction = Ranking(models, 'Abstraction')
    abstraction.add_order_increasing(models)
    scope = Ranking(models, 'Scope')
    scope.add_order_decreasing(models)
    return resolution, abstraction, scope


def multiple_scope():
    """
    Resolution, abstraction, and scope, with purely increasing
    resolution and abstraction and multiple opinions on scope: one
    equivalent and one decreasing

    Notes
    -----
    Orders should typically be added using Order from model_rankings
    and Ranking.add_order, shortcut methods are used here for these
    simplified cases

    Returns
    -------
    individual_ranking
    """
    resolution = Ranking(models, 'Resolution')
    resolution.add_order_increasing(models)
    abstraction = Ranking(models, 'Abstraction')
    abstraction.add_order_increasing(models)
    scope = Ranking(models, 'Scope')
    scope.add_order_fixed(models)
    scope.add_order_decreasing(models)
    return resolution, abstraction, scope


# notional_rankings = no_scope()
# notional_rankings = fixed_scope()
# notional_rankings = decreasing_scope()
notional_rankings = multiple_scope()
combined = Ranking.combine(notional_rankings, name='notional')
if plot:
    # Plot fidelity KDE distributions
    combined_models = combined.plot_scores(
        cdf=cdf, cmap=cmap, title=False, legend=True, use_ids=True)
    if tight:
        plt.tight_layout()
    if save:
        plt.savefig('notional-dists.{}'.format(save))
    # Plot stacked bar charts of fidelity probabilities
    combined.plot_score_bars(cmap=cmap)

if __name__ == '__main__':
    plt.show()
