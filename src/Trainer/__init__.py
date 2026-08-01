"""
Trainer classes for LaSDI.

`Trainer` owns shared training state, normalization, logging, checkpointing, and diagnostics.
`First_Order_Rollout` and `Second_Order_Rollout` train first- and second-order latent dynamics
with reconstruction, coefficient, latent residual, and rollout losses. `First_Order_Weak` and
`Second_Order_Weak` reuse the rollout trainer structure for weak-form latent-dynamics residuals.
"""

from    .Trainer                import  Trainer;
from    .First_Order_Rollout    import  First_Order_Rollout;
from    .First_Order_Weak       import  First_Order_Weak;
from    .Second_Order_Rollout   import  Second_Order_Rollout;
from    .Second_Order_Weak      import  Second_Order_Weak;

__all__ = [    "Trainer",
               "First_Order_Rollout",
               "First_Order_Weak",
               "Second_Order_Rollout",
               "Second_Order_Weak"];
