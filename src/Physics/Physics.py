# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  logging;

import  numpy;
import  torch


# Setup logger
LOGGER : logging.Logger = logging.getLogger(__name__);


# -------------------------------------------------------------------------------------------------
# Physics class 
# -------------------------------------------------------------------------------------------------

class Physics:
    # Physical space dimension
    dim         : int           = -1;

    # The fom solution can be vector valued. If it is, then qdim specifies the dimensionality of 
    # the fom solution at each point. If the solution is scalar valued, then qdim = -1. 
    qdim        : int           = -1;
    
    # grid_size is the shape of the grid nd-array.
    grid_size   : list[int]     = [];
    
    # the shape of the solution nd-array. This is just the qgrid_size with the qdim prepended onto 
    # it.
    qgrid_size  : list[int]     = [];
    
    '''
        numpy nd-array, assuming the shape of:
        - 1d: (space_dim[0],)
        - 2d: (2, space_dim[0], space_dim[1])
        - 3d: (3, space_dim[0], space_dim[1], space_dim[2])
        - higher dimension...
    '''
    x_grid      : numpy.ndarray = numpy.array([]);

    # the number of time steps, as a positive integer.
    nt          : int           = -1;

    # time step size. assume constant for now. 
    dt          : float         = -1.;

    # time grid in numpy 1d array. 
    t_grid      : numpy.ndarray = numpy.array([]);
    
    # list of parameter names to parse parameters.
    param_name_list : list[str] = None;



    def __init__(self, cfg : dict, param_name_list : list[str] = None) -> None:
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

        param_name_list: A list of strings. There should be one list item for each parameter. The 
        i'th element of this list should be a string housing the name of the i'th parameter.

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """
        
        self.param_name_list = param_name_list
        return
    


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

        A list of d-dimensional numpy.ndarray objects of shape self.grid_size. Here, 
        d = len(self.grid_size). The i'th element of this list holds the initial state of the i'th 
        time derivative of the FOM state.
        """

        raise RuntimeError("Abstract method Physics.initial_condition!")
    


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

        A list of (ns + 2)-dimensional torch.Tensor objects of shape (1, nt, nx[0], .. , 
        nx[ns - 1]), where nt is the number of points along the temporal grid and nx = 
        self.grid_size specifies the number of grid points along the axes in the spatial grid.
        """

        raise RuntimeError("Abstract method Physics.solve!")
    


    def export(self) -> dict:
        """
        This function should return a dictionary that houses self's state. I
        """
        raise RuntimeError("Abstract method Physics.export!")
    


    def generate_solutions(self, params : numpy.ndarray) -> list[torch.Tensor]:
        """
        Given 2d-array of params, generate solutions of size params.shape[0]. params.shape[1] must 
        match the required size of parameters for the specific physics.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param: a 2d numpy.ndarray object of shape (np, n), where np is the number of combinations 
        of parameters we want to test and n denotes the number of parameters in self's initial 
        condition function.

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        A list of torch.Tensor objects of shape (np, nt, nx[0], .. , nx[ns - 1]), where nt is the 
        number of points along the temporal grid and nx = self.grid_size specifies the number of 
        grid points along the axes in the spatial grid. The i'th element of this list should hold
        the i'th time derivative of the FOM solutions.
        """

        # Make sure we have a 2d grid of parameter values.
        assert(params.ndim == 2)
        n_param : int = len(params)

        # Report
        LOGGER.info("Generating %d samples" % n_param)

        # Cycle through the parameters.
        X_Train : list[torch.Tensor] = [];
        for k, param in enumerate(params):
            # Solve the underlying equation using the current set of parameter values.
            new_X : list[torch.Tensor] = self.solve(param);

            # Now, add this solution to the set of solutions.
            assert(new_X[0].shape[0] == 1) # should contain one parameter case.
            if (len(X_Train) == 0):
                X_Train = new_X;
            else:
                for i in range(len(new_X)):
                    X_Train[i] = torch.cat([X_Train[i], new_X[i]], dim = 0);

            LOGGER.info("%d/%d complete" % (k + 1, n_param));

        # All done!
        return X_Train;



    def residual(self, Xhist : numpy.ndarray) -> tuple[numpy.ndarray, float]:
        """
        The user should write an instance of this method for their specific Physics sub-class.
        This function should compute the PDE residual (difference between the left and right hand 
        side of of the underlying physics equation when we substitute in the solution in Xhist).


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Xhist: A (ns + 1)-dimensional numpy.ndarray object of shape self.grid_size  = (nt, nx[0], 
        ... , nx[ns - 1]), where nt is the number of points along the temporal grid and nx = 
        self.grid_size specifies the number of grid points along the axes in the spatial grid. 
        The i,j(0), ... , j(ns - 1) element of this array should hold the value of the solution at 
        the i'th time step and the spatial grid point with index (j(0), ... , j(ns - 1)).


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        A two element tuple. The first is a numpy.ndarray object holding the residual on the 
        spatial and temporal grid. The second should be a float holding the norm of the residual.
        """

        raise RuntimeError("Abstract method Physics.residual!")
