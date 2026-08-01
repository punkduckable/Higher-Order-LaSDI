"""
Sampling strategies for LaSDI active learning.

`Sampler` defines the base greedy active-learning interface and data-generation hooks.
`FOM_Rollout` selects new parameters using intrusive full-order rollout error estimates.
`FOM_Variance` selects parameters using decoded full-order variance from sampled ROM rollouts.
"""

from    .Sampler                import  Sampler;
from    .FOM_Rollout            import  FOM_Rollout;
from    .FOM_Variance           import  FOM_Variance, get_FOM_max_std;

__all__ = [    "Sampler",
               "FOM_Rollout",
               "FOM_Variance",
               "get_FOM_max_std"];
