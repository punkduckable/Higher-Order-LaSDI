"""
Latent-dynamics models for LaSDI.

`LatentDynamics` defines the shared coefficient, weak-form, and rollout interface. `SINDy` and
`SwitchSINDy` define first-order affine latent ODEs, including switching dynamics. `DampedSpring`
defines second-order latent dynamics, and the `_weak` variants use weak-form residual losses.
"""

from    .LatentDynamics         import  LatentDynamics;
from    .DampedSpring           import  DampedSpring;
from    .DampedSpring_weak      import  DampedSpring_weak;
from    .Interpolatable         import  InterpolatableLatentDynamics;
from    .SINDy                  import  SINDy;
from    .SINDy_weak             import  SINDy_weak;
from    .SwitchSINDy            import  SwitchSINDy;
from    .SwitchSINDy_weak       import  SwitchSINDy_weak;
from    .Weak                   import  WeakLatentDynamics;

__all__ = [    "LatentDynamics",
               "SINDy",
               "SINDy_weak",
               "SwitchSINDy",
               "SwitchSINDy_weak",
               "DampedSpring",
               "DampedSpring_weak",
               "WeakLatentDynamics",
               "InterpolatableLatentDynamics"];
