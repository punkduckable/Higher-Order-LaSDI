"""
Latent-coefficient interpolation.

`Interpolate` defines an abstract Interpolate class. 

`GPInterpolate` is an Interpolate sub-class that uses Gaussian Processes to define the coefficient
posterior distributions.
coefficients. 
"""

from    .GaussianProcess        import  GPInterpolate;
from    .Interpolate            import  Interpolate;

__all__ = [    "GPInterpolate",
               "Interpolate"];
