"""
Latent-coefficient interpolation and rollout helpers.

`GaussianProcess` fits, evaluates, and samples independent Gaussian processes for latent-dynamics
coefficients. `Interpolate` wraps those Gaussian processes around native coefficient dictionaries.
`Rollouts` evaluates ROM trajectories using posterior-mean or sampled interpolated coefficients.
"""

from    .GaussianProcess        import  fit_gps, eval_gp, sample_coefs;
from    .Interpolate            import  Interpolate;
from    .Rollouts               import  Mean_Rollout, Sample_Rollouts;

__all__ = [    "fit_gps",
               "eval_gp",
               "sample_coefs",
               "Interpolate",
               "Mean_Rollout",
               "Sample_Rollouts"];
