# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  logging;

import  numpy;
import  torch;


# Logger setup.
LOGGER : logging.Logger = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# LatentDynamics base class
# -------------------------------------------------------------------------------------------------

class LatentDynamics:
    # Class variables
    n_z             : int   = -1;       # Dimensionality of the latent space
    n_coefs         : int   = -1;       # Number of coefficients in the latent space dynamics
    n_IC            : int   = -1;       # Number of initial conditions to define the initial latent state.
    Uniform_t_Grid  : bool  = False;    # Is there an h such that the i'th frame is at t0 + i*h? Or is the spacing between frames arbitrary?

    # TODO(kevin): do we want to store coefficients as an instance variable?
    coefs   : torch.Tensor  = torch.Tensor([]);



    def __init__(   self, 
                    n_z             : int,
                    coef_norm_order : str | float,  
                    Uniform_t_Grid  : bool) -> None:
        """
        Initializes a LatentDynamics object. Each LatentDynamics object needs to have a 
        dimensionality (n_z), a number of time steps, a model for the latent space dynamics, and 
        set of coefficients for that model. The model should describe a set of ODEs in 
        \mathbb{R}^{n_z}. These ODEs should contain a set of unknown coefficients. We learn those 
        coefficients using the calibrate function. Once we have learned the coefficients, we can 
        solve the corresponding set of ODEs forward in time using the simulate function.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        n_z: The number of dimensions in the latent space, where the latent dynamics takes place.

        coef_norm_order: A string or float specifying which norm we want to use when computing
        the coefficient loss. We pass as the "p" argument to torch.norm. 

        Uniform_t_Grid: A boolean which, if True, specifies that for each parameter value, the 
        times corresponding to the frames of the solution for that parameter value will be 
        uniformly spaced. In other words, the first frame corresponds to time t0, the second to 
        t0 + h, the k'th to t0 + (k - 1)h, etc (note that h may depend on the parameter value, but
        it needs to be constant for a specific parameter value). The value of this setting 
        determines which finite difference method we use to compute time derivatives. 
        """

        # Set class variables.
        self.n_z                : int           = n_z;
        self.coef_norm_order    : str | float   = coef_norm_order; 
        self.Uniform_t_Grid     : bool          = Uniform_t_Grid;

        # There must be at least one latent dimension and there must be at least 1 time step.
        assert(self.n_z > 0);

        # All done!
        return;
    


    def calibrate(  self, 
                    Latent_States   : list[list[torch.Tensor]], 
                    t_Grid          : list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The user must implement this class on any latent dynamics sub-class. Each latent dynamics 
        object should implement a parameterized model for the dynamics in the latent space. A 
        Latent_Dynamics object should pair each combination of parameter values with a set of 
        coefficients in the latent space. Using those parameters, we compute loss functions (one 
        characterizing how well the left and right hand side of the latent dynamics match, another
        specifies the norm of the coefficient matrix). 

        This function computes the optimal coefficients and the losses, which it returns.

        Specifically, this function should take in a sequence (or sequences) of latent states and a
        set of time grids, t_Grid, which specify the time associated with each Latent State Frame.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_States: An n_param (number of parameter combinations we want to calibrate) element
        list. The i'th list element should be an n_IC element list whose j'th element is a 2d numpy 
        array of shape (n_t(i), n_z) whose p, q element holds the q'th component of the j'th 
        derivative of the latent state during the p'th time step (whose time value corresponds to 
        the p'th element of t_Grid) when we use the i'th combination of parameter values. 
        
        t_Grid: An n_param element list of 1d torch.Tensor objects. The i'th element should be a 
        1d tensor of length n_t(i) whose j'th element holds the time value corresponding to the 
        j'th frame when we use the i'th combination of parameter values.


        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Three variables: coefs, loss_sindy, and loss_coef. 
        
        coefs holds the coefficients. It is a matrix of shape (n_train, n_coef), where n_train 
        is the number of parameter combinations in the training set and n_coef is the number of 
        coefficients in the latent dynamics. The i,j entry of this array holds the value of the 
        j'th coefficient when we use the i'th combination of parameter values.

        loss_sindy holds the total SINDy loss. It is a single element tensor whose lone entry holds
        the sum of the SINDy losses across the set of combinations of parameters in the training 
        set. 

        loss_coef is a single element tensor whose lone element holds the sum of the L1 norms of 
        the coefficients across the set of combinations of parameters in the training set.
        """

        raise RuntimeError('Abstract function LatentDynamics.calibrate!');
    


    def simulate(   self,
                    coefs   : numpy.ndarray         | torch.Tensor, 
                    IC      : list[numpy.ndarray]   | list[torch.Tensor],
                    t_Grid  : list[torch.Tensor]    | list[numpy.ndarray]) -> list[list[numpy.ndarray]]  | list[list[torch.Tensor]]:
        """
        Time integrates the latent dynamics from multiple initial conditions for each combination
        of coefficients in coefs. 
 
        

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        coefs: A dimensional numpy.ndarray or torch.Tensor object of shape (n_param, n_coef). The 
        i'th row should represent the optimal set of coefficients when we use the i'th combination 
        of parameter values. We inductively call simulate on each row of coefs. 

        IC: A n_IC element list of numpy.ndarray or torch.Tensor objects. If 
        coefs.shape = (n_param, n_coef)), then each element of IC should have shape 
        (n_param, n, n_z). In this case, IC[d][i, j, :] represents the j'th initial condition for 
        the d'th derivative of the latent state when we use the i'th combination of parameter 
        values.

        t_Grid: An n_param element list of torch.Tensor or numpy.ndarray objects. If coefs has 
        shape (n_param, n_coef) and each element of IC has shape (n_param, n, n_z), then the 
        i'th element of t_Grid should be a 2d numpy ndarray object of shape (n, n_t(i)) whose 
        j, k entry specifies the k'th time value for the j'th initial condition when we use the 
        i'th set of coefficients. Each row of each array should be in ascending order.

        

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------        
        
        A list of n_param lists. The i'th list item is a list of length n_IC. 
        
        If coefs has shape (n_param, n_coefs), the j'th entry of the i'th list should be a 3d 
        array of shape (n_t(i), n, n_z), where n_t(i) is the number of time steps in the i'th 
        combination of parameter values. The p, q, r entry of this array should hold the r'th 
        component of the q'th sample of the p'th time step of the j'th derivative latent state when 
        we sample the coefficients from the posterior distribution for the i'th combination of 
        parameter values.
        """

        raise RuntimeError('Abstract function LatentDynamics.simulate!');
    


    def export(self) -> dict:
        param_dict = {'n_z'             : self.n_z, 
                      'n_coefs'         : self.n_coefs, 
                      'n_IC'            : self.n_IC,
                      'Uniform_t_Grid'  : self.Uniform_t_Grid};
        return param_dict;



    # SINDy does not need to load parameters.
    # Other latent dynamics might need to.
    def load(self, dict_ : dict) -> None:
        assert(self.n_z             == dict_['n_z']);
        assert(self.n_coefs         == dict_['n_coefs']);
        assert(self.n_IC            == dict_['n_IC']);
        assert(self.coef_norm_order == dict_['coef_norm_order']);
        assert(self.Uniform_t_Grid  == dict_['Uniform_t_Grid']);
        return;
    