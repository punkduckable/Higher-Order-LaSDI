"""
Latent-coefficient interpolation.

`GaussianProcess` fits, evaluates, and samples independent Gaussian processes for latent-dynamics
coefficients. `Interpolate` wraps those Gaussian processes around native coefficient dictionaries.
"""

from    .GaussianProcess        import  fit_gps, eval_gp, sample_coefs;
from    .Interpolate            import  Interpolate;

__all__ = [    "fit_gps",
               "eval_gp",
               "sample_coefs",
               "Interpolate"];
