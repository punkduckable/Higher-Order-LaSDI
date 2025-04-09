# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  logging;

import  numpy;
import  torch;


# Setup logger
LOGGER : logging.Logger = logging.getLogger(__name__);


# -------------------------------------------------------------------------------------------------
# Physics class 
# -------------------------------------------------------------------------------------------------

class Physics:
    # spatial dimension of the problem domain.
    spatial_dim : int       = -1;
    
    # The FOM solution can be vector valued. If it is, then qdim specifies the dimensionality of 
    # the FOM solution at each point. If the solution is scalar valued, then qdim = -1. 
    qdim        : int       = -1;

    # The shape of each frame of a FOM solution to this equation. This is the shape of the objects
    # we will put into our autoencoder.
    Frame_Shape : list[int] = [];

    # A dictionary housing the configuration parameters for the Physics object.
    config      : dict      = {};
    
    # list of parameter names to parse parameters.
    param_names : list[str] = None;




    def __init__(self, config : dict, param_names : list[str] = None) -> None:
        """
        A Physics object acts as a wrapper around a solver for a particular equation. The initial 
        condition in that function can have named parameters. Each physics object should have a 
        solve method to solve the underlying equation for a given set of parameters, and an 
        initial condition function to recover the equation's initial condition for a specific set 
        of parameters.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        cfg: A dictionary housing the settings for the Physics object. This should be the "physics"
        sub-dictionary of the main configuration file. 

        param_names: A list of strings. There should be one list item for each parameter. The i'th 
        element of this list should be a string housing the name of the i'th parameter.

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """
        
        self.config         = config;
        self.param_names    = param_names;
        return;
    


    def initial_condition(self, param : numpy.ndarray) -> list[numpy.ndarray]:
        """
        The user should write an instance of this method for their specific Physics sub-class.
        It should evaluate and return the initial condition along the spatial grid.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param: A 1d numpy.ndarray object holding the value of self's parameters (necessary to 
        specify the IC).
        

        -------------------------------------------------------------------------------------------
        Returns 
        -------------------------------------------------------------------------------------------

        A list of numpy.ndarray objects of shape self.Frame_Shape. The i'th element of this list 
        holds the initial state of the i'th time derivative of the FOM state.
        """

        raise RuntimeError("Abstract method Physics.initial_condition!");
    


    def solve(self, param : numpy.ndarray) -> list[torch.Tensor]:
        """
        The user should write an instance of this method for their specific Physics sub-class.
        This function should solve the underlying equation when the IC uses the parameters in 
        param.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param: A 1d numpy.ndarray object with two elements corresponding to the values of the 
        initial condition parameters.

                
        -------------------------------------------------------------------------------------------
        Returns 
        -------------------------------------------------------------------------------------------

        A two element tuple: X, t_Grid.
         
        X is a 2 element list holding the displacement and velocity of the FOM solution when we use
        param. Each element is a 3d torch.Tensor object of shape (1, n_t, self.Frame_Shape), where 
        n_t is the number of time steps when we solve the FOM using param for the IC parameters.

        t_Grid is a 1d numpy.ndarray object whose i'th element holds the i'th time value at which
        we have an approximation to the FOM solution (the time value associated with X[0, i, ...]).
        """

        raise RuntimeError("Abstract method Physics.solve!");
    


    def export(self) -> dict:
        """
        Returns a dictionary housing self's internal state. You can use this dictionary to 
        effectively serialize self.
        """

        dict_ : dict = {'config'        : self.config, 
                        'param_names'   : self.param_names};
        return dict_;
    


    def generate_solutions(self, params : numpy.ndarray) -> tuple[list[list[torch.Tensor]], list[numpy.ndarray]]:
        """
        Given 2d-array of params, generate solutions of size params.shape[0]. params.shape[1] must 
        match the required size of parameters for the specific physics.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param: a 2d numpy.ndarray object of shape (n_param, n_p), where n_param is the number of 
        combinations of parameters we want to test and n_p denotes the number of parameters in 
        self's initial condition function.

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        A two element tuple: X, t_Grid.

        X is an n_param element list whose i'th element is an n_IC element list whose j'th element
        is a torch.Tensor object of shape (n_t(i), n_x[0], ... , n_x[ns- 1]) holding the j'th 
        derivative of the FOM solution for the i'th combination of parameter values. Here, n_IC is 
        the number of initial conditions needed to specify the IC, n_param is the number of rows 
        in param, n_t(i) is the number of time steps we used to generate the solution with the 
        i'th combination of parameter values (the length of the i'th element of t_Grid).

        t_Grid is a list whose i'th element is a 1d numpy array housing the time steps from the 
        solution to the underlying equation when we use the i'th combination of parameter values.
        """

        # Make sure we have a 2d grid of parameter values.
        assert(params.ndim == 2);
        n_params : int = len(params);

        # Report
        LOGGER.info("Generating solution for %d parameter combinations" % n_params);

        # Cycle through the parameters.
        X       : list[list[torch.Tensor]]  = [];
        t_Grid  : list[numpy.ndarray]       = [];
        for j in range(n_params):
            param   = params[j, :];

            # Solve the underlying equation using the current set of parameter values.
            new_X, new_t_Grid = self.solve(param);

            # Now, add this solution to the set of solutions.
            assert(new_X[0].shape[0] == 1) # should contain one parameter case.
            X.append(new_X);
            t_Grid.append(new_t_Grid);

            LOGGER.info("%d/%d complete" % (j + 1, n_params));

        # All done!
        return X, t_Grid;



    def residual(self, Xhist : numpy.ndarray, t_Grid : numpy.ndarray) -> tuple[numpy.ndarray, float]:
        """
        The user should write an instance of this method for their specific Physics sub-class.
        This function should compute the PDE residual (difference between the left and right hand 
        side of of the underlying physics equation when we substitute in the solution in Xhist).


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Xhist: A (ns + 1)-dimensional numpy.ndarray object of shape (n_t, self.Frame_Shape), where 
        n_t is the length of t_Grid.

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        A two element tuple. The first is a numpy.ndarray object holding the residual on the 
        spatial and temporal grid. The second should be a float holding the norm of the residual.
        """

        raise RuntimeError("Abstract method Physics.residual!");
