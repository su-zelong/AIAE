# AIAE
**This is a personal reproduction of an AIAE paper**

## Note:
The current is still an unstable version, up to this point the AIAE-model has been constructed; the corresponding loss; dataset partitioning, etc., but the learning process of GCN=AIW in the upperEncoder leads to a loss that does not converge, and attempts have been made to add Normalization or other regularization methods. Also the choice of optimizer may be another major conflict. See parameter.py for details of specific parameters

## requirements
torch=2.4.0+cuda118
torch_geometric=2.5.3
scikit-learn=1.5.1

## This reproduction is for learning purposes only.

Original link: https://doi.org/10.1016/j.knosys.2024.111583
