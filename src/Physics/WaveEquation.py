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
from    wave_equation                   import  Simulate, Initial_Displacement, Initial_Velocity;


LOGGER : logging.Logger = logging.getLogger(__name__);




# -------------------------------------------------------------------------------------------------
# WaveEquation class
# -------------------------------------------------------------------------------------------------

class WaveEquation(Physics):
    def __init__(self, config : dict, param_names : list[str]) -> None:
        """
        Initialize a WaveEquation object. This class acts as a wrapper around the MFEM-based solver 
        implemented in ``wave_equation.py`` within the ``PyMFEM`` sub-directory. The solver models 
        the propagation of a wave in a two dimensional domain:

                (d^2/dt^2)u(t, X) = c^2*laplacian(u(t, X))

        Subject to the following initial conditions:
        
            u(0, (x, y))        = exp(-k*(x^2 + y^2))
            (d/dt)u(0, (x, y))  = 0
            

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        config : dict
            Dictionary housing the settings for the WaveEquation object. This should be the 
            ``physics`` sub-dictionary of the configuration file.
        
        param_names : list[str], optional
            Names of parameters appearing in the initial condition. The wave equation has two 
            parameters, "c" and "k". "k" impacts the rate of decay of the initial condition (see
            initial_condition) while "c" changes the governing equation (see above).

        
        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        None
        """

        # Run checks
        assert isinstance(param_names, list),   "type(param_names) = %s" % str(type(param_names));
        assert len(param_names) == 2,           "len(param_names) = %d" % len(param_names);
        assert 'c' in param_names,              "param_names = %s" % str(param_names);
        assert 'k' in param_names,              "param_names = %s" % str(param_names);
        assert('WaveEquation' in config);

        # Run a short simulation to determine the frame shape and positions.
        U, DtU, X, T                        = Simulate(t_Grid = numpy.linspace(0, 0, 2), VisIt = False);

        # Call the super class initializer.
        super().__init__(config         = config,
                         spatial_dim    = 2,
                         X_Positions    = numpy.copy(X),
                         Frame_Shape    = list(U.shape[1:]),
                         param_names    = param_names,
                         Uniform_t_Grid = False,
                         n_IC           = 2);

        # Determine which index corresponds to c (wave speed) and which to k (decay rate in the IC).
        self.c_idx  : int   = self.param_names.index('c');
        self.k_idx  : int   = self.param_names.index('k');        
        return;



    def initial_condition(self, param : numpy.ndarray) -> list[numpy.ndarray]:
        """
        Evaluate the initial condition of the wave equation at the positions stored in 
        ``self.X_Positions``. For the default problem considered in the MFEM example, the initial 
        conditions are
             
                    u(0, (x, y))    = exp(-k*(x^2 + y^2))
            (d/dt)  u(0, (x, y))    = 0

        We initialie and call the Initial_Displacement and Initial_Velocity classes from the 
        wave_equation.py file in the PyMFEM sub-directory.


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
        assert isinstance(param, numpy.ndarray), "type(param) = %s" % str(type(param));
        assert len(param.shape) == 1,            "len(param.shape) = %d" % len(param.shape);
        assert param.shape[0]   == self.n_p,     "param.shape = %s, self.n_p = %d" % (str(param.shape), self.n_p);
        assert self.X_Positions is not None,     "self.X_Positions is None";

        # Fetch the parameters.
        k : float = param[self.k_idx];

        # Initialize the initial condition classes.
        initial_displacement : Initial_Displacement = Initial_Displacement(k = k);
        initial_velocity     : Initial_Velocity     = Initial_Velocity(k = k);

        # Evaluate the initial condition.
        u0 : numpy.ndarray = initial_displacement.EvalValue(self.X_Positions);     # shape = (1, N)
        v0 : numpy.ndarray = initial_velocity.EvalValue(self.X_Positions);         # shape = (1, N)

        # Return the initial conditions.
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

        assert isinstance(param, numpy.ndarray), "type(param) = %s" % str(type(param));
        assert len(param.shape) == 1,            "len(param.shape) = %d" % len(param.shape);
        assert param.shape[0]   == self.n_p,     "param.shape = %s, self.n_p = %d" % (str(param.shape), self.n_p);

        # Set up the t_Grid.
        n_t     : int           = self.config['WaveEquation']['n_t'];
        t_max   : float         = self.config['WaveEquation']['t_max']; 
        t_Grid  : numpy.ndarray = numpy.linspace(0, t_max, n_t, dtype = numpy.float32);
        if(self.Uniform_t_Grid == False):
            r               : float = 0.2*(t_Grid[1] - t_Grid[0]);
            t_adjustments           = numpy.random.uniform(low = -r, high = r, size = (n_t - 2));
            t_Grid[1:-1]            = t_Grid[1:-1] + t_adjustments;

        # Solve the PDE using the external MFEM script.
        U, DtU, _, Times = Simulate(k                   = param[self.k_idx], 
                                    c                   = param[self.c_idx], 
                                    Positions           = self.X_Positions, 
                                    t_Grid              = t_Grid, 
                                    VisIt               = False, 
                                    serialization_steps = 1);

        X       : list[torch.Tensor] = [torch.Tensor(U), torch.Tensor(DtU)];
        t_Grid  : torch.Tensor       = torch.Tensor(Times);
        return X, t_Grid;
