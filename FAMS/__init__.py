"""
TODO
----
- Split Ranking and CombinedRanking?
  - Ability to look at probabilities based on one fidelity metric
  - Clarification since now two different types of Ranking
- MultiRanking: Look at multiple responses
  - MultiRanking would be multiple generic metrics
  - Combined ModelRanking still multi-attribute since response + cost
  - MultiModelRanking would be multiple outputs + cost
- Portfolios
  - Can look at model portfolios with ModelRanking and MultiModelRanking
  - MultiModelRanking would just be different scoring method
  - For generic items, could look at portfolios for Ranking, but might
    just be additive (potentially with compatibility matrix)
  - MultiRanking would involve MADM/TOPSIS and be more interesting
"""