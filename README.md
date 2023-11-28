# FAMS : Fidelity Assessment for Model Selection

Author : *Adam Cox, PhD*

Contact : adam.cox@asdl.gatech.edu

## Capabilities

### Expert-driven ranking of items
This was originally developed the rank the fidelity of models via 
ranked ordering in terms of the resolution, abstraction, and scope of the 
given models.

This has been expanded to provide the capability to perform a similar 
assessment for a list of generic items, such as a list of technologies.

#### Input
For each metric, the users provide a set of ranked orders to designate 
whether each item is better/worse/equivalent to each other item

#### Outputs
- Main output: For each item, a score based on the probability that item is 
  the best of the set
- Other outputs:
  - Probability of second, third, ..., last
  - Notional distribution using KDE based on expert-provided samples

### For Models

#### Model data-driven correction of fidelity scores

#### Fidelity - Efficiency Multi-Attribute Scoring and Decision-Making

**For more information see [Adam's dissertation](https://smartech.gatech.edu/handle/1853/61283)**

## Changelog

### v1.1.1: Minor bug fix version
See commit message, multiple minor bug fixes

### v1.1: Method generalization, usability features
Create generalized Ranking object and differentiate from ModelRanking object in order to be able to use expert-driven data processing to assess more generic items such as technologies in addition to just models.

Other updates:
- Updated parameters such as description, category
- Capability to store/load rankings in JSON files, including processed probabilities
- Updated examples and plotting
- New TOPSIS method
- Progress bars
