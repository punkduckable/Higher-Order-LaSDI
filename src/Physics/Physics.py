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
    spatial_dim :    int            = -1;
    
    # The shape of each frame of a FOM solution to this equation. This is the shape of the objects
    # we will put into our autoencoder. If there is no structure to the spatial positions of the 
    # nodes in each solution frame, then this may be a single element list specifying the number
    # of nodes. On the other hand, if the nodes are organized into a grid with k axes, then this 
    # could be a k-element list whose i'th element specifies the size of the i'th axis. If the 
    # solution is vector-valued, the dimensionality of the solution vectors should be the leading 
    # element of Frame_Shape.
    Frame_Shape     : list[int]     = None;

    # At each frame, we evaluate the solution at a fixed number of positions in the spatial portion
    # of the problem domain. This array should hold the coordinates of those positions. It may be 
    # organized as a grid of coordinates, list of coordinates, or something else. Different 
    # sub-classes will format this differently. We only use this for plotting purposes, so the 
    # exact shape doesn't really matter. 
    X_Positions     : numpy.ndarray = None;

    # A dictionary housing the configuration parameters for the Physics object.
    config          : dict          = None;
    
    # list of parameter names to parse parameters.
    param_names     : list[str]     = None;

    # The number of parameters. i.e., the length of param_names.
    n_p             : int           = -1;

    # If true, then we can assume that for each parameter value, the t_Grid for that parameter 
    # value has uniformly sized time steps (t_Grid[i + 1] - t_Grid[i] = dt is the same for each i).
    # This allows us to use higher order finite difference schemes, for instance. 
    Uniform_t_Grid  : bool          = False;

    # How many derivatives of the initial state do we need to fully specify the initial condition
    # of the Physics?
    n_IC            : int           = -1;




    def __init__(self, config : dict, param_names : list[str] = None, Uniform_t_Grid : bool = False) -> None:
        """
        A Physics object acts as a wrapper around a solver for a particular equation. The initial 
        condition in that function can have named parameters. Each physics object should have a 
        solve method to solve the underlying equation for a given set of parameters, and an 
        initial condition function to recover the equation's initial condition for a specific set 
        of parameters.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        config: dict 
            A dictionary housing the settings for the Explicit object. This should be the "physics" 
            sub-dictionary of the configuration file. 

        param_names: list[str], len = 2
            i'th element be a string housing the name of the i'th parameter. 

        Uniform_t_Grid: bool
            If True, then for each parameter value, the times corresponding to the frames of the 
            solution for that parameter value will be uniformly spaced. In other words, the first 
            frame corresponds to time t0, the second to t0 + h, the k'th to t0 + (k - 1)h, etc 
            (note that h may depend on the parameter value, but it needs to be constant for a 
            specific parameter value). The value of this setting determines which finite difference 
            method we use to compute time derivatives. 


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Checks
        assert(isinstance(config, dict));
        assert(isinstance(param_names, list));
        assert(isinstance(Uniform_t_Grid, bool));

        self.n_p            : int   = len(param_names);
        for i in range(self.n_p):
            assert(isinstance(param_names[i], str));
    

        # Setup.
        self.config         : dict      = config;
        self.param_names    : list[str] = param_names;
        self.Uniform_t_Grid : bool      = Uniform_t_Grid;
        return;
    


    def initial_condition(self, param : numpy.ndarray) -> list[numpy.ndarray]:
        """
        The user should write an instance of this method for their specific Physics sub-class.
        It should evaluate and return the initial condition along the spatial grid.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param: 1d numpy.ndarray, shape = (n_p)
            i'th element holds the value of self's i'th parameter. 
        

        -------------------------------------------------------------------------------------------
        Returns 
        -------------------------------------------------------------------------------------------

        u0: list[numpy.ndarray], len = self.n_IC
            i'th element has shape shape self.Frame_Shape and holds initial state of the i'th time 
            derivative of the FOM state.
        """

        raise RuntimeError("Abstract method Physics.initial_condition!");
    


    def solve(self, param : numpy.ndarray) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        The user should write an instance of this method for their specific Physics sub-class.
        This function should solve the underlying equation when the IC uses the parameters in 
        param.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param: numpy.ndarray, shape = (n_p)
           Holds the value of one combination of parameters for the initial condition. Here, n_p 
           is the number of parameters in self's initial condition function.

                
        -------------------------------------------------------------------------------------------
        Returns 
        -------------------------------------------------------------------------------------------

        X, t_Grid.
         
        X : list[torch.Tensor], len = n_IC
            i'th element holds the i'th derivative of the FOM solution when we use param to define 
            the FOM. Each element is a torch.Tensor object of shape (n_t, self.Frame_Shape), where 
            n_t is the number of time steps when we solve the FOM using param.

        t_Grid : torch.Tensor, shape = (n_t)
            i'th element holds the i'th time value at which we have an approximation to the FOM 
            solution (the time value associated with X[0, i, ...]).
        """

        raise RuntimeError("Abstract method Physics.solve!");
    


    def export(self) -> dict:
        """
        Returns a dictionary housing self's internal state. You can use this dictionary to 
        effectively serialize self.
        """

        dict_ : dict = {'config'            : self.config, 
                        'param_names'       : self.param_names,
                        'Uniform_t_Grid'    : self.Uniform_t_Grid};
        return dict_;
    


    def generate_solutions(self, params : numpy.ndarray) -> tuple[list[list[torch.Tensor]], list[torch.Tensor]]:
        """
        For each row of params, solve the underlying physics using that row to define the FOM.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param : numpy.ndarray, shape = (n_param, n_p)
            i, j entry specifies the value of the j'th parameter in the i'th combination of 
            parameters. n_param is the number of parameters in self's initial condition function.

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        X, t_Grid.

        X : list[list[torch.Tensor]], len = n_param
            i'th element is an n_IC element list whose j'th element is a torch.Tensor object of 
            shape (n_t(i), self.Frame_Shape) holding the j'th derivative of the FOM solution for 
            the i'th combination of parameter values. Here, n_IC is the number of initial 
            conditions needed to specify the IC, n_param is the number of rows in param, n_t(i) is 
            the number of time steps we used to generate the solution with the i'th combination of 
            parameter values (the length of the i'th element of t_Grid).

        t_Grid : list[torch.Tensor], len = n_param
            i'th element is a 1d torch.Tensor of shape (n_t(i)) housing the time steps from the 
            solution to the underlying equation when we use the i'th combination of parameter 
            values to define FOM.
        """

        # Make sure we have a 2d grid of parameter values.
        assert(params.ndim == 2);
        n_params : int = len(params);

        # Report
        LOGGER.info("Generating solution for %d parameter combinations" % n_params);

        # Cycle through the parameters.
        X       : list[list[torch.Tensor]]  = [];
        t_Grid  : list[torch.Tensor]        = [];
        for j in range(n_params):
            param   = params[j, :];
            LOGGER.info("Generating solution for parameter %d, %s" % (j, str(param)));

            # Solve the underlying equation using the current set of parameter values.
            new_X, new_t_Grid = self.solve(param);

            # Now, add this solution to the set of solutions.
            X.append(new_X);
            t_Grid.append(new_t_Grid);

            LOGGER.info("%d/%d complete" % (j + 1, n_params));

        # All done!
        return X, t_Grid;
