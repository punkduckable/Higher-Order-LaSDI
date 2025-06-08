# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

# Add the main directory to the search path.
import logging;
import  os;
import  sys;
PyMFEM_Path     : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "PyMFEM"));
sys.path.append(PyMFEM_Path);

import  numpy;
from    scipy.special                   import  erfc;
import  torch;

from    Physics                         import  Physics;
from    advection                       import  Simulate;


LOGGER : logging.Logger = logging.getLogger(__name__);




# -------------------------------------------------------------------------------------------------
# Advection class
# -------------------------------------------------------------------------------------------------

class Advection(Physics):
    def __init__(self, config : dict, param_names : list[str] = None) -> None:
        """
        Initialize an Advection object. This class acts as a wrapper around the MFEM-based solver 
        implemented in ``advection.py`` within the ``PyMFEM`` sub-directory. The solver models 
        the transport of a scalar quantity on a two dimensional domain.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        config : dict
            Dictionary housing the settings for the Advection object. This should be the 
            ``physics`` sub-dictionary of the configuration file.
        
        param_names : list[str], optional
            Names of parameters appearing in the initial condition. The advection model has two 
            parameters w (which specifies the rotation speed of the velocity field) and k (which 
            specifies the frequency of peaks in the initial condition).

        
        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        None
        """

        # Run checks
        assert(isinstance(param_names, list));
        assert(len(param_names) == 2);
        assert('k' in param_names);
        assert('w' in param_names);
        assert('Advection' in config);

        # Call the super class initializer.
        super().__init__(config         = config,
                         param_names    = param_names,
                         Uniform_t_Grid = False);

        # Run a short simulation to determine the frame shape and positions.
        Sol, X, T, bb_min, bb_max           = Simulate(t_final = 0, VisIt = False);
        self.Frame_Shape    : list[int]     = list(Sol.shape[1:]);
        self.X_Positions    : numpy.ndarray = numpy.copy(X);            # shape = (2, N)
        self.bb_min         : numpy.ndarray = numpy.copy(bb_min);
        self.bb_max         : numpy.ndarray = numpy.copy(bb_max);
        LOGGER.debug("Frame shape: %s" % str(self.Frame_Shape));

        # Since there are two spatial dimensions, set spatial_dim accordingly.
        self.spatial_dim    : int           = 2;
        self.n_IC           : int           = 1;

        # Make sure the config dictionary is actually for the advection model.
        self.k_idx  : int   = self.param_names.index('k');
        self.w_idx  : int   = self.param_names.index('w');        
        return;



    def initial_condition(self, param : numpy.ndarray) -> list[numpy.ndarray]:
        """
        Evaluate the initial condition of the advection equation at the positions stored in 
        ``self.X_Positions``. For the default problem considered in the MFEM example, the initial 
        state is defined by
             
            u_0(x, t) = sin(pi * k * x) * sin(pi * k * y)

        Here, k = param[0] and w = param[1] (w defines the governing equation but is unused in the 
        initial condition).

        Note: The initial condition is defined on the unit square.
        

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param : numpy.ndarray, shape = (self.n_p)
            A two element array holding the values of the k and w parameters.


                
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        u0 : list[numpy.ndarray], len = 1
            A single element list whose element has shape (1, N) holding the value of the initial 
            condition at each of the N spatial locations.
        """

        # Checks
        assert(isinstance(param, numpy.ndarray));
        assert(len(param.shape) == 1);
        assert(param.shape[0]   == self.n_p);
        assert(self.X_Positions is not None);

        # Bounding box used to non-dimensionalize the coordinates.
        center : numpy.ndarray = (self.bb_min + self.bb_max) / 2.0;
        X      : numpy.ndarray = 2.0 * (self.X_Positions - center.reshape(-1, 1)) / (self.bb_max - self.bb_min).reshape(-1, 1);

        # Evaluate the initial condition.
        u0     : numpy.ndarray = numpy.sin(numpy.pi * param[self.k_idx] * X[0]) * numpy.sin(numpy.pi * param[self.k_idx] * X[1]);
        return [u0.reshape(1, -1)];



    def solve(self, param : numpy.ndarray) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Solve the advection equation. This function simply calls the MFEM solver ``Simulate`` and 
        packages its output so that it conforms to the :class:`Physics` API.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param : numpy.ndarray, shape = (self.n_p)
            A two element array holding the values of the k and w parameters.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        X, t_grid 

        X : list[torch.Tensor], len = 1
            A one element list containing a tensor of shape (n_t, *self.Frame_Shape) with the 
            solution trajectory.

        t_grid : torch.Tensor
            A one dimensional tensor of the corresponding times.
        """

        assert(isinstance(param, numpy.ndarray));
        assert(len(param.shape) == 1);
        assert(param.shape[0]   == self.n_p);

        # Solve the PDE using the external MFEM script.
        Sol, _, Times, _, _ = Simulate(k = param[self.k_idx], w = param[self.w_idx], Positions = self.X_Positions, VisIt = False);

        X       : list[torch.Tensor] = [torch.Tensor(Sol)];
        t_Grid  : torch.Tensor       = torch.Tensor(Times);
        return X, t_Grid;
