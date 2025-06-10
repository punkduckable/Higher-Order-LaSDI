# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

# Add the main directory to the search path.
import  logging;
import  os;
import  sys;
PyMFEM_Path     : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "PyMFEM"));
sys.path.append(PyMFEM_Path);

import  numpy;
import  torch;

from    Physics                         import  Physics;
from    klein_gordan                    import  Simulate;


LOGGER : logging.Logger = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# WaveEquation class
# -------------------------------------------------------------------------------------------------

class KleinGordan(Physics):
    def __init__(self, config : dict, param_names : list[str] = None) -> None:
        """
        Initialize a KleinGordan object. This class acts as a wrapper around the MFEM-based solver 
        implemented in ``klein_gordan.py`` within the ``PyMFEM`` sub-directory. We solve the 
        Klein-Gordon equation in a two dimensional spatial domain:

            (d^2/dt^2)u(t X) - c^2*laplacian(u(t, X)) + m^2*u(t, X) = 0

        with the following initial condition:

            u(0, (x, y))        = exp(-k*(x^2 + y^2)) * sin(pi*w*x) * sin(pi*w*y)
            (du/dt)(0, (x, y))  = 0


                      
        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        config : dict
            Dictionary housing the settings for the WaveEquation object. This should be the 
            ``physics`` sub-dictionary of the configuration file.
        
        param_names : list[str], optional
            Names of parameters appearing in the initial condition. This should include "w" and
            "m" parameters. "w" controls the frequency of peaks in the IC (see initial_condition) 
            while "m" impacts the governing equation (see above). "c" in the equation is fixed.

        
        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        None
        """

        # Run checks
        assert(isinstance(param_names, list));
        assert(len(param_names) == 2);
        assert('w' in param_names);
        assert('m' in param_names);
        assert('KleinGordan' in config);

        # Call the super class initializer.
        super().__init__(config         = config,
                         param_names    = param_names,
                         Uniform_t_Grid = False);

        # Run a short simulation to determine the frame shape and positions.
        U, DtU, X, T                        = Simulate(t_final = 0, VisIt = False);
        self.Frame_Shape    : list[int]     = list(U.shape[1:]);
        self.X_Positions    : numpy.ndarray = numpy.copy(X);            # shape = (2, N)    
        LOGGER.debug("Frame shape: %s" % str(self.Frame_Shape));

        # Since there are two spatial dimensions, set spatial_dim accordingly.
        self.spatial_dim    : int           = 2;
        self.n_IC           : int           = 2;

        # Determine which index corresponds to c (wave speed) and which to k (decay rate in the IC).
        self.w_idx  : int   = self.param_names.index('w');
        self.m_idx  : int   = self.param_names.index('m');        
        return;



    def initial_condition(self, param : numpy.ndarray) -> list[numpy.ndarray]:
        """
        Evaluate the initial condition of the advection equation at the positions stored in 
        ``self.X_Positions``. For the default problem considered in the MFEM example, the initial 
        state is defined by
             
            u(0, (x, y))        = exp(-k*(x^2 + y^2)) * sin(pi*w*x) * sin(pi*w*y)
            (du/dt)(0, (x, y))  = 0

        Here, w = param[w_idx] and c = param[c_kdx], and k = 1.0. 
        
        Note 1: c is unused in the IC but defines the wave speed in wave equation, 
        
            d^2u/dt^2 = c^2 * d^2u/dx^2,
       
        Note 2: The initial condition is defined on a star-shaped domain.
        

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param : numpy.ndarray, shape = (self.n_p)
            A two element array holding the values of the k and c parameters.


                
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        u0 : list[numpy.ndarray], len = 2   
            A two element list whose i'th element is a numpy.ndarray of shape (1, N) holding the 
            value of the i'th time derivative of the initial condition at each of the N spatial 
            locations. 
        """

        # Checks
        assert(isinstance(param, numpy.ndarray));
        assert(len(param.shape) == 1);
        assert(param.shape[0]   == self.n_p);
        assert(self.X_Positions is not None);

        # Evaluate the initial condition.
        k       : float             = 1.0;
        w       : float             = param[self.w_idx];
        X       : numpy.ndarray     = self.X_Positions;     # (2, N_x)
        norm2   : float             = numpy.sum(numpy.square(X), axis = 0);
        u0      : numpy.ndarray     = numpy.multiply(numpy.exp(-k*numpy.sum(numpy.square(X), axis = 0)), numpy.sin(numpy.pi*w*X[0, :]) * numpy.sin(numpy.pi*w*X[1, :])).reshape(1, -1);
        v0      : numpy.ndarray     = numpy.zeros_like(u0);
     
        return [u0, v0];



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

        X : list[torch.Tensor], len = 2 
            A two element list containing a tensor of shape (n_t, *self.Frame_Shape) with the 
            solution trajectory. The first element is the solution u(x, t) and the second element 
            is the solution du/dt(x, t).

        t_grid : torch.Tensor, shape = (n_t)
            A one dimensional tensor of the corresponding times.
        """

        assert(isinstance(param, numpy.ndarray));
        assert(len(param.shape) == 1);
        assert(param.shape[0]   == self.n_p);

        # Solve the PDE using the external MFEM script.
        U, DtU, _, Times = Simulate(w = param[self.w_idx], m = param[self.m_idx], Positions = self.X_Positions, VisIt = True);

        X       : list[torch.Tensor] = [torch.Tensor(U), torch.Tensor(DtU)];
        t_Grid  : torch.Tensor       = torch.Tensor(Times);
        return X, t_Grid;
