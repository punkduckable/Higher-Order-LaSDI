# ------------------------------------------------------------------------------
# Imports and Setup
# ------------------------------------------------------------------------------

# Add the main directory to the search path.
import  os
import  sys
PyMFEM_Path     : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "PyMFEM"))
sys.path.append(PyMFEM_Path)

import  numpy
from    scipy.special           import  erfc
import  torch

from    Physics                         import  Physics
from    advection                      import  Simulate


# ------------------------------------------------------------------------------
# Advection class
# ------------------------------------------------------------------------------

class Advection(Physics):
    def __init__(self, config : dict, param_names : list[str] = None) -> None:
        """
        Initialize an Advection object. This class acts as a wrapper around the
        MFEM-based solver implemented in ``advection.py`` within the ``PyMFEM``
        sub-directory. The solver models the transport of a scalar quantity on a
        two dimensional domain.

        Parameters
        ----------
        config : dict
            Dictionary housing the settings for the Advection object. This should
            be the ``physics`` sub-dictionary of the configuration file.
        param_names : list[str], optional
            Names of parameters appearing in the initial condition. The advection
            example has no parameters so this should be an empty list.

        Returns
        -------
        None
        """

        # Default parameter names to an empty list
        if param_names is None:
            param_names = []

        # Checks
        assert(len(param_names) == 0)

        # Call the super class initializer.
        super().__init__(config         = config,
                         param_names    = param_names,
                         Uniform_t_Grid = False)

        # Run a short simulation to determine the frame shape and positions.
        Sol, X, T                    = Simulate(t_final = 0)
        self.Frame_Shape    : list[int]     = list(Sol.shape[1:])
        self.X_Positions    : numpy.ndarray = numpy.copy(X)

        # Since there are two spatial dimensions, set spatial_dim accordingly.
        self.spatial_dim    : int           = X.shape[0]
        self.n_IC           : int           = 1

        # Make sure the config dictionary is actually for the advection model.
        assert('Advection' in config)
        return


    def initial_condition(self, param : numpy.ndarray) -> list[numpy.ndarray]:
        """
        Evaluate the initial condition of the advection equation at the positions
        stored in ``self.X_Positions``. For the default problem considered in the
        MFEM example, the initial state is defined by

        .. math::
            u_0(x,y) = \frac{1}{16} \prod_{s\in\{\pm1\}} \operatorname{erfc}(w (X_0-s r_x))
                            \operatorname{erfc}(s w (X_1-r_y)),
        where ``X`` denotes the coordinates mapped to ``[-1,1]^2`` and ``w`` is
        ``10``.

        Parameters
        ----------
        param : numpy.ndarray, shape = (self.n_p)
            Parameter values describing the initial condition. For this physics
            model the array is empty, but it is included for API compatibility.

        Returns
        -------
        list[numpy.ndarray]
            A single element list whose element has shape ``(1, N)`` holding the
            value of the initial condition at each of the ``N`` spatial
            locations.
        """

        # Checks
        assert(isinstance(param, numpy.ndarray))
        assert(len(param.shape) == 1)
        assert(param.shape[0]   == self.n_p)
        assert(self.X_Positions is not None)

        # Bounding box used to non-dimensionalize the coordinates.
        bb_min = numpy.array([-1.0, -1.0])
        bb_max = numpy.array([ 1.0,  1.0])
        center = (bb_min + bb_max) / 2.0

        X = 2.0 * (self.X_Positions - center[:, None]) / (bb_max - bb_min)[:, None]

        rx  : float = 0.45
        ry  : float = 0.25
        cx  : float = 0.0
        cy  : float = -0.2
        w   : float = 10.0

        u0  : numpy.ndarray = (erfc(w * (X[0] - cx - rx)) *
                               erfc(-w * (X[0] - cx + rx)) *
                               erfc(w * (X[1] - cy - ry)) *
                               erfc(-w * (X[1] - cy + ry))) / 16.0

        return [u0.reshape(1, -1)]


    def solve(self, param : numpy.ndarray) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Solve the advection equation. This function simply calls the MFEM solver
        ``Simulate`` and packages its output so that it conforms to the
        :class:`Physics` API.

        Parameters
        ----------
        param : numpy.ndarray, shape = (self.n_p)
            Array of parameter values. It is unused for this physics model but is
            included for API compatibility.

        Returns
        -------
        tuple
            ``(X, t_grid)`` where ``X`` is a one element list containing a tensor
            of shape ``(n_t, *self.Frame_Shape)`` with the solution trajectory and
            ``t_grid`` is a one dimensional tensor of the corresponding times.
        """

        assert(isinstance(param, numpy.ndarray))
        assert(len(param.shape) == 1)
        assert(param.shape[0]   == self.n_p)

        # Solve the PDE using the external MFEM script.
        Sol, _, Times = Simulate()

        X       : list[torch.Tensor] = [torch.Tensor(Sol)]
        t_Grid  : torch.Tensor       = torch.Tensor(Times)
        return X, t_Grid
