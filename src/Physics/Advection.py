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
from    advection                       import  Simulate, Initial_Displacement;


LOGGER : logging.Logger = logging.getLogger(__name__);




# -------------------------------------------------------------------------------------------------
# Advection class
# -------------------------------------------------------------------------------------------------

class Advection(Physics):
    def __init__(self, config : dict, param_names : list[str] = []) -> None:
        r"""
        Initialize an Advection object. This class acts as a wrapper around the MFEM-based solver 
        implemented in ``advection.py`` within the ``PyMFEM`` sub-directory. The solver models 
        the transport of a scalar quantity on a two dimensional domain. Specifically, it solves 
        the following PDE:

            (d/dt)u(t, X) + v(X) \cdot \nabla(u(t, X)) = 0

        with the following initial condition:
            
            u(0, (x, y))    = exp(-k*(x~^2 + y~^2)) * sin(pi*w*x~) * sin(pi*w*y~))
        
        Here, x~ and y~ are rescaled coordinates. See the u0_coeff class in ./MFEM/acvetion.py for
        details. 
            

            
        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        config : dict
            Dictionary housing the settings for the Advection object. This should be the 
            ``physics`` sub-dictionary of the configuration file.
        
        param_names : list[str], optional
            Names of parameters appearing in the initial condition. The advection model has two 
            parameters g (which specifies the rotation speed of the velocity field) and w (which 
            specifies the frequency of peaks in the initial condition).

        
        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        None
        """

        # Run checks
        assert(isinstance(param_names, list));
        assert(len(param_names) == 2);
        assert('w' in param_names);
        assert('g' in param_names);
        assert('Advection' in config);


        # Run a short simulation to determine the frame shape and positions.
        Sol, X, T, bb_min, bb_max           = Simulate(t_Grid = numpy.linspace(0, 0.01, 2), VisIt = False);
        self.bb_min         : numpy.ndarray = numpy.copy(bb_min);
        self.bb_max         : numpy.ndarray = numpy.copy(bb_max);

        # Call the super class initializer.
        super().__init__(spatial_dim    = 2,
                         Frame_Shape    = list(Sol.shape[1:]),
                         X_Positions    = numpy.copy(X),
                         config         = config,
                         param_names    = param_names,
                         Uniform_t_Grid = config['Advection']['uniform_t_grid'],
                         n_IC           = 1);

        # Record the default value of k (for the initial condition).
        self.k              : float         = 1.0;

        # Make sure the config dictionary is actually for the advection model.
        self.w_idx  : int   = self.param_names.index('w');
        self.g_idx  : int   = self.param_names.index('g');        
        return;



    def initial_condition(self, param : numpy.ndarray) -> list[numpy.ndarray]:
        """
        Evaluate the initial condition of the advection equation at the positions stored in 
        ``self.X_Positions``. For the default problem considered in the MFEM example, the initial 
        state is defined by
             
            u(0, (x, y)) = exp(-w(x^2 + y^2)) * sin(pi*k*x) * sin(pi*k*y)

        Here, w  = 1.0 is fixed while k is derived from params. 
        

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

        # Fetch the parameters.
        w : float = param[self.w_idx];

        # Initialize the initial condition classes.
        initial_displacement : Initial_Displacement = Initial_Displacement(bb_min = self.bb_min, bb_max = self.bb_max, k = self.k, w = w);

        # Evaluate the initial condition.
        u0     : numpy.ndarray  = initial_displacement.EvalValue(self.X_Positions); # shape = (2, N)
        return [u0];



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

        # Set up the t_Grid.
        n_t     : int           = self.config['Advection']['n_t'];
        t_max   : float         = self.config['Advection']['t_max']; 
        t_Grid  : numpy.ndarray = numpy.linspace(0, t_max, n_t, dtype = numpy.float32);
        if(self.Uniform_t_Grid == False):
            r               : float = 0.2*(t_Grid[1] - t_Grid[0]);
            t_adjustments           = numpy.random.uniform(low = -r, high = r, size = (n_t - 2));
            t_Grid[1:-1]            = t_Grid[1:-1] + t_adjustments;

        # Solve the PDE using the external MFEM script.
        Sol, _, Times, _, _ = Simulate( w                   = param[self.w_idx], 
                                        k                   = self.k, 
                                        g                   = param[self.g_idx], 
                                        Positions           = self.X_Positions, 
                                        t_Grid              = t_Grid, 
                                        VisIt               = False,
                                        serialization_steps = 1);
        X       : list[torch.Tensor] = [torch.Tensor(Sol)];
        t_Grid  : torch.Tensor       = torch.Tensor(Times);
        return X, t_Grid;
