# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

# Add the main directory to the search path.
import  os;
import  sys;
PyMFEM_Path     : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), "PyMFEM"));
sys.path.append(PyMFEM_Path);

import  numpy;
import  torch;

from    Physics                         import  Physics;
from    nonlinear_elasticity            import  Simulate;



# -------------------------------------------------------------------------------------------------
# Explicit class
# -------------------------------------------------------------------------------------------------

class NonlinearElasticity(Physics):    
    def __init__(self, config : dict, param_names : list[str] = None) -> None:
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
        assert(len(param_names) == 2);
        assert('s' in param_names);
        assert('K' in param_names);

        # Call the super class initializer.
        super().__init__(config         = config, 
                         param_names    = param_names, 
                         Uniform_t_Grid = False);

        # Since there are 2 spatial dimensions, dim is 2. 
        self.spatial_dim    : int           = 2;
        self.n_IC           : int           = 2; 

        # Next, we need to setup X_Positions and Frame_Shape. Doing this is a bit tricky, because 
        # the solver actually picks both quantities. Specifically, in this case, Frame_Shape is 
        # [2, N_Nodes, 2] and X_Positions has shape [N_Nodes, 2]. The issue is that we have to run 
        # a simulation to get N_Nodes. We run a simulation with a final time of zero; this prompts
        # the code to generate the mesh and nodes, but not to solve for anything
        D, V, X, T                          = Simulate(t_final = 0);        # D, V have shape (Nt, 2, N_Nodes)
        self.Frame_Shape    : list[int]     = list(D.shape[1:]);
        self.X_Positions    : numpy.ndarray = numpy.copy(X);

        # Make sure the config dictionary is actually for the explicit physics model.
        assert('NonlinearElasticity' in config);

        # Determine which index corresponds to s and which to K (simulate accepts a two element
        # array holding s and K. We need to know which element corresponds to K and which to s).
        self.s_idx  : int   = self.param_names.index('s');
        self.K_idx  : int   = self.param_names.index('K');
        return;
    


    def initial_condition(self, param : numpy.ndarray) -> list[numpy.ndarray]:
        """
        Evaluates the initial condition at the points in self.X_Positions. In this case,

        x((x, y), 0)        =   (x, y)
        v((x, y), 0)        =   (-s*x^2, s*x^2 (8.0 - x))

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
            i'th element has shape (2, N_Nodes) (where N_Nodes = X.shape[0]) and holds the i'th 
            derivative of the initial state when we use param to define the FOM.
        
        """

        # Checks.
        assert(isinstance(param, numpy.ndarray));
        assert(self.X_Positions is not None);
        assert(len(param.shape) == 1);
        assert(param.shape[0]   == self.n_p);


        # Fetch s
        s   : float             = param[0];

        # Compute the initial condition and return!
        X   : numpy.ndarray     = numpy.copy(self.X_Positions);         # Shape = (2, N_Nodes)
        u0  : numpy.ndarray     = X;
        v0  : numpy.ndarray     = numpy.empty_like(u0);
        v0[0, :]    = -s*numpy.multiply(X[0, :], X[0, :]);
        v0[1, :]    = s*numpy.multiply(X[0, :], X[0, :])*(8.0*numpy.ones_like(X[0, :]) - X[0, :]);

        # All done!
        return [u0, v0];



    def solve(self, param : numpy.ndarray) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Solves the following PDE when s = param[0]:

            (dv/dt)(X, t)   = H(x(X, t)) + S v(X, t), 
            (dx/dt)(X, t)   = v(X, t),

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

        assert(isinstance(param, numpy.ndarray));
        assert(len(param.shape) == 1);
        assert(param.shape[0]   == self.n_p);
        
        # Solve the PDE!
        D, V, _, T  = Simulate(theta = param[self.s_idx], bulk_modulus = param[self.K_idx]);

        # All done!
        X       : list[torch.Tensor]    = [torch.Tensor(D), torch.Tensor(V)];
        t_Grid  : torch.Tensor          = torch.Tensor(T);

        return X, t_Grid;
    