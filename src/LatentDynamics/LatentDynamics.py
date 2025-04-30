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

        n_z : int
            The number of dimensions in the latent space, where the latent dynamics takes place.

        coef_norm_order : float, 'inf', 'fro'
            Specifies which norm we want to use when computing the coefficient loss. We pass this 
            as the "p" argument to torch.norm. If it's a float, coef_norm_order = p \in \mathbb{R}, 
            then we use the corresponding l^p norm. If it is "inf" or "fro", we use the infinity 
            or Frobenius norm, respectively. 

        Uniform_t_Grid : bool 
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

        Latent_States : list[list[torch.Tensor]], len = n_param
            The i'th list element should be an n_IC element list whose j'th element is a 2d numpy 
            array of shape (n_t(i), n_z) whose p, q element holds the q'th component of the j'th 
            derivative of the latent state during the p'th time step (whose time value corresponds 
            to the p'th element of t_Grid) when we use the i'th combination of parameter values. 
        
        t_Grid : list[troch.Tensor], len = n_param
            The i'th element should be a 1d tensor of shape (n_t(i)) whose j'th element holds the 
            time value corresponding to the j'th frame when we use the i'th combination of 
            parameter values.


        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        coefs, loss_sindy, loss_coef. 
        
        coefs : torch.Tensor, shape = n_train, n_coef
            Here, n_train is the number of parameter combinations in the training set and n_coef 
            is the number of coefficients in the latent dynamics. The i,j entry of this array 
            holds the value of the j'th coefficient when we use the i'th combination of parameter 
            values.

        loss_sindy : torch.Tensor, shape = [] 
            A 0-dimensional tensor whose lone element holds holds the sum of the SINDy losses 
            across the set of combinations of parameters in the training set. 

        loss_coef : torch.Tensor, shape = []
            A 0-dimensional tensor whose lone element holds the sum of the L1 norms of the 
            coefficients across the set of combinations of parameters in the training set.
        """

        raise RuntimeError('Abstract function LatentDynamics.calibrate!');
    


    def simulate(   self,
                    coefs   : numpy.ndarray             | torch.Tensor, 
                    IC      : list[list[numpy.ndarray]] | list[list[torch.Tensor]],
                    t_Grid  : list[numpy.ndarray]       | list[torch.Tensor]) -> list[list[numpy.ndarray]]  | list[list[torch.Tensor]]:
        """
        Time integrates the latent dynamics from multiple initial conditions for each combination
        of coefficients in coefs. 
 

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        coefs : numpy.ndarray or torch.Tensor, shape = (n_param, n_coef)
            i'th row represents the optimal set of coefficients when we use the i'th combination 
            of parameter values. We inductively call simulate on each row of coefs. 

        IC : list[list[numpy.ndarray]] or list[list[torch.Tensor]], len = n_param
            i'th element is an n_IC element list whose j'th element is a 2d numpy.ndarray or 
            torch.Tensor object of shape (n(i), n_z). Here, n(i) is the number of initial 
            conditions (for a fixed set of coefficients) we want to simulate forward using the i'th 
            set of coefficients. Further, n_z is the latent dimension. If you want to simulate a 
            single IC, for the i'th set of coefficients, then n(i) == 1. IC[i][j][k, :] should hold 
            the k'th initial condition for the j'th derivative of the latent state when we use the 
            i'th combination of parameter values. 

        t_Grid : list[numpy.ndarray] or list[torch.Tensor], len = n_param
            i'th entry is a 2d numpy.ndarray or torch.Tensor whose shape is either (n(i), n_t(i)) 
            or shape (n_t(i)). The shape should be 2d if we want to use different times for each 
            initial condition and 1d if we want to use the same times for all initial conditions. 
        
            In the former case, the j,k array entry specifies k'th time value at which we solve for 
            the latent state when we use the j'th initial condition and the i'th set of 
            coefficients. Each row should be in ascending order. 
        
            In the latter case, the j'th entry should specify the j'th time value at which we solve 
            for each latent state when we use the i'th combination of parameter values.
        

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------        
        
        Z : list[list[numpy.ndarray]] or list[list[torch.Tensor]], len = n_parm
            i'th element is a list of length n_IC whose j'th entry is a 3d array of shape 
            (n_t(i), n(i), n_z). The p, q, r entry of this array should hold the r'th component of 
            the p'th frame of the j'th tine derivative of the solution to the latent dynamics when 
            we use the q'th initial condition for the i'th combination of parameter values.
        """

        raise RuntimeError('Abstract function LatentDynamics.simulate!');
    


    def export(self) -> dict:
        param_dict = {'n_z'             : self.n_z, 
                      'n_coefs'         : self.n_coefs, 
                      'n_IC'            : self.n_IC,
                      'coef_norm_order' : self.coef_norm_order,
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
    