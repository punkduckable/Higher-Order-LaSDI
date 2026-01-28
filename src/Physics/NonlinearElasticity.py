# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

# Add the main directory to the search path.
import  os;
import  sys;
import  logging

from torch._library.fake_class_registry import FakeScriptMethod;
PyMFEM_Path     : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "PyMFEM"));
sys.path.append(PyMFEM_Path);

import  numpy;
import  torch;

from    Physics                         import  Physics;
from    nonlinear_elasticity            import  Simulate, Initial_Displacement, Initial_Velocity;

LOGGER : logging.Logger = logging.getLogger(__name__);




# -------------------------------------------------------------------------------------------------
# Explicit class
# -------------------------------------------------------------------------------------------------

class NonlinearElasticity(Physics):    
    def __init__(self, config : dict, param_names : list[str]) -> None:
        """
        This is the initializer the NonlinearElasticity class. This class acts as a wrapper around
        an MFEM script that solves the following PDE from non-linear elasticity:

            (dv/dt)(x, t) = H(x) + Sv(x, t)
            
        Here, H(x) is a hyper-elastic model and S is a viscosity operator. The script
        "nonlinear_elasticity.py" in the PyMFEM sub-directory solves this problem.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        config : dict
            A dictionary housing the settings for the NonlinearElasticity object. This should 
            be the "physics" sub-dictionary of the configuration file. 

        param_names : list[str]
            A list of strings. There should be one list item for each parameter. The i'th element 
            of this list should be a string housing the name of the i'th parameter. 

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Checks.
        assert isinstance(param_names, list), "type(param_names) = %s" % str(type(param_names));
        assert len(param_names) == 2,         "len(param_names) = %d, param_names = %s" % (len(param_names), str(param_names));
        assert 's'  in param_names,           "param_names = %s" % str(param_names);
        assert 'mu' in param_names,           "param_names = %s" % str(param_names);
        assert('NonlinearElasticity' in config);

        # Next, we need to setup X_Positions and Frame_Shape. Doing this is a bit tricky, because 
        # the solver actually picks both quantities. Specifically, in this case, Frame_Shape is 
        # [2, num_positions] and X_Positions has shape [2, num_positions]. The issue is that we have 
        # to run a simulation to get num_positions. We run a simulation with a final time of zero; 
        # this prompts the code to generate the mesh and nodes, but not to solve for anything
        D, V, X, T                          = Simulate(t_Grid = numpy.linspace(0, 0, 2), VisIt = False);     # D, V have shape (Nt, 2, num_positions)


        # Call the super class initializer.
        super().__init__(config         = config, 
                         spatial_dim    = 2,
                         X_Positions    = numpy.copy(X),
                         Frame_Shape    = list(D.shape[1:]),
                         param_names    = param_names, 
                         Uniform_t_Grid = False,
                         n_IC           = 2);


        # Determine which index corresponds to s and which to mu (simulate accepts a two element
        # array holding s and mu. We need to know which element corresponds to mu and which to s).
        self.s_idx  : int   = self.param_names.index('s');
        self.mu_idx : int   = self.param_names.index('mu');
        return;
    


    def initial_condition(self, param : numpy.ndarray) -> list[numpy.ndarray]:
        """
        Evaluates the initial condition at the points in self.X_Positions. In this case,

            d((x, y), 0)        = (x, y)
            v((x, y), 0)        = (-s*x^2, s*x^2 (8.0 - x))

        Here, s = param[0].


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param : numpy.ndarray, shape = (self.n_p)
            The single element corresponding to the values of the w and a parameters. self.a_idx and 
            self.w_idx tell us which index corresponds to which variable.


        -------------------------------------------------------------------------------------------
        Returns 
        -------------------------------------------------------------------------------------------

        X0 : list[numpy.ndarray], len = self.n_IC
            i'th element has shape (2, num_positions), where num_positions = self.X_Positions.shape[1]. 
            This array holds the i'th derivative of the initial state when we use param to define 
            the FOM.
        """

        # Checks.
        assert isinstance(param, numpy.ndarray), "type(param) = %s" % str(type(param));
        assert self.X_Positions is not None,     "self.X_Positions is None";
        assert len(param.shape) == 1,            "len(param.shape) = %d" % len(param.shape);
        assert param.shape[0]   == self.n_p,     "param.shape = %s, self.n_p = %d" % (str(param.shape), self.n_p);

        # Fetch s.
        s   : float             = param[self.s_idx];

        # Initialize the initial condition classes.
        initial_displacement : Initial_Displacement = Initial_Displacement(dim = self.spatial_dim, s = s);
        initial_velocity     : Initial_Velocity     = Initial_Velocity(dim = self.spatial_dim, s = s);

        # Compute the initial condition and return!
        u0 : numpy.ndarray = initial_displacement.EvalValue(self.X_Positions);
        v0 : numpy.ndarray = initial_velocity.EvalValue(self.X_Positions);

        # All done!
        return [u0, v0];



    def solve(self, param : numpy.ndarray) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Solves the following PDE when s = param[0]:

            (dv/dt)(t, X)   = H(x(t, X)) + S v(t, X), 
            (dx/dt)(t, X)   = v(t, X),

        with

            x((x, y), 0)         =  (x, y)
            v((x, y), 0)         =  (-s*x^2, s*x^2 (8.0 - x))


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param: numpy.ndarray, shape = (2)
            Holds the values of the s and . self.a_idx and self.w_idx tell us which 
            index corresponds to which variable.


        -------------------------------------------------------------------------------------------
        Returns 
        -------------------------------------------------------------------------------------------

        X, t_Grid

        X : list[torch.Tensor], len = 2
            i'th element has shape = (n_t, self.Frame_Shape), holds the i'th derivative of the 
            FOM solution when we use param to define the FOM. Specifically, the [j, ...] sub-array 
            of the returned array holds the i'th derivative of the FOM solution at t_Grid[j].

        t_Grid : torch.Tensor, shape = (n_t)
            i'th element holds the i'th time value at which we have an approximation to the FOM 
            solution (the time value associated with X[i, ...]).
        """

        assert isinstance(param, numpy.ndarray), "type(param) = %s" % str(type(param));
        assert len(param.shape) == 1,            "len(param.shape) = %d" % len(param.shape);
        assert param.shape[0]   == self.n_p,     "param.shape = %s, self.n_p = %d" % (str(param.shape), self.n_p);
        
        # Set up the t_Grid.
        n_t     : int           = self.config['NonlinearElasticity']['n_t'];
        t_max   : float         = self.config['NonlinearElasticity']['t_max']; 
        t_Grid  : numpy.ndarray = numpy.linspace(0, t_max, n_t, dtype = numpy.float32);
        if(self.Uniform_t_Grid == False):
            r               : float = 0.2*(t_Grid[1] - t_Grid[0]);
            t_adjustments           = numpy.random.uniform(low = -r, high = r, size = (n_t - 2));
            t_Grid[1:-1]            = t_Grid[1:-1] + t_adjustments;

        # Solve the PDE!
        D, dD_dt, _, T  = Simulate( s                   = param[self.s_idx], 
                                    shear_modulus       = param[self.mu_idx], 
                                    Positions           = self.X_Positions, 
                                    t_Grid              = t_Grid, 
                                    VisIt               = False, 
                                    serialization_steps = 1);
    
        # All done!
        X       : list[torch.Tensor]    = [torch.Tensor(D), torch.Tensor(dD_dt)];
        t_Grid  : torch.Tensor          = torch.Tensor(T);

        return X, t_Grid;
    