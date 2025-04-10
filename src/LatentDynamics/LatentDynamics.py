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
    n_z     : int           = -1;       # Dimensionality of the latent space
    n_t      : int           = -1;      # Number of time steps when solving the latent dynamics
    n_coefs : int           = -1;       # Number of coefficients in the latent space dynamics
    n_IC    : int           = -1;       # Number of initial conditions to define the initial latent state.

    # TODO(kevin): do we want to store coefficients as an instance variable?
    coefs   : torch.Tensor  = torch.Tensor([]);



    def __init__(self, n_z_ : int, n_t_ : int) -> None:
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

        n_z_: The number of dimensions in the latent space, where the latent dynamics takes place.
        """

        # Set class variables.
        self.n_z    : int   = n_z_;
        self.n_t    : int   = n_t_;

        # There must be at least one latent dimension and there must be at least 1 time step.
        assert(self.n_z > 0);
        assert(self.n_t  > 0);

        # All done!
        return;
    


    def calibrate(  self, 
                    Latent_States   : list[list[torch.Tensor]]  | list[torch.Tensor], 
                    t_Grid          : list[numpy.ndarray]       | numpy.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        Latent_States: Either a list of lists of 2d tensors, or a list of 2d tensors. 
        
        If Latent_States is a list of lists, then t_Grid should be a list of 1d numpy.ndarray 
        object. In this case, Latent_States should be a list of length n_param (number of parameter 
        combinations we want to calibrate). The i'th list element should be an n_IC element list 
        whose j'th element is a 2d numpy array of shape (n_t(i), n_z) whose p, q element holds the 
        q'th component of the j'th derivative of the latent state during the p'th time step (whose 
        time value corresponds to the p'th element of t_Grid) when we use the i'th combination of 
        parameter values. 
        
        if Latent_States is a list of numpy arrays, then t_Grid should be a 1d numpy arrays. In 
        this case, Latent_States should be a list of length n_IC. The j'th element of this list 
        should be an numpy ndarray of shape (n_t(i), n_z) whose p, q element holds the q'th 
        component of the j'th derivative of the latent state during the p'th time step (whose 
        time value corresponds to the p'th element t_Grid).
        
        t_Grid: Either a list of 1d numpy ndarray objects or a 1d numpy ndarray object. See the 
        description of Latent_States for details. 

        
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
                    t_Grid  : list[numpy.ndarray]   | numpy.ndarray) -> list[list[numpy.ndarray]]  | list[list[torch.Tensor]]:
        """
        Time integrates the latent dynamics from multiple initial conditions for each combination
        of coefficients in coefs. 
 

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        coefs: Either a one or two dimensional numpy.ndarray or torch.Tensor objects. If it has 
        one dimension, then coefs represents a is a flattened copy of array of latent dynamics 
        coefficients that calibrate returns. If coefs has two dimensions, then it should have shape 
        (n_param, n_coef) and it's i'th row should represent the optimal set of coefficients when 
        we use the i'th combination of parameter values. In this case, we inductively call simulate 
        on each row of coefs. 

        IC: A n_IC element list of numpy.ndarray or torch.Tensor objects. These arrays should 
        have either two or three dimensions. If coefs has one dimension, then each element of IC 
        should have shape (n, n_z) where n represents the number of initial conditions (for a fixed
        set of coefficients) we simulate forward in time and n_z is the latent dimension. If you 
        want to simulate a single IC, then n == 1. If coefs has two dimensions (specifically if 
        coefs.shape = (n_param, n_coef)), then each element of IC should have shape (n_param, n, 
        n_z). In this case, IC[d][i, j, :] represents the j'th initial condition for the d'th 
        derivative of the latent state when we use the i'th combination of parameter values.

        t_Grid: Either a list of 2d numpy ndarrays or a 2d numpy ndarray object. If coefs has two 
        dimensions, then t_Grid should be a list whose i'th entry is a 2d numpy ndarray object 
        of shape (n, n_t(i)) whose j, k entry specifies the k'th time value for the j'th initial
        condition when we use the i'th set of coefficients. If coefs has just one dimension, then 
        t_Grid should be a 2d numpy ndarray object of shape (n, n_t(i)) of whose j, k entry 
        specifies the k'th time value for the j'th initial condition.
        
        In both cases, the array elements should be in ascending order.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------        
        
        A list of n_param lists. The i'th list item is a list of length n_IC. 
        
        If coefs has two dimensions and shape (n_param, n_coefs), the j'th entry of the i'th 
        list should be a 3d array of shape (n_t(i), n, n_z), where n_t(i) is the number of time 
        steps in the i'th combination of parameter values. The p, q, r entry of this array should 
        hold the r'th component of the q'th sample of the p'th time step of the j'th derivative 
        latent state when we sample the coefficients from the posterior distribution for the i'th 
        combination of parameter values.

        If coefs has one dimension, the the returned item a single element list whose lone element 
        is an n_IC element list whose j'th element is an array of shape (n_t, n, n_z). The p,q,r 
        element of the returned arrays should house the r'th component of the q'th sample of the 
        p'th frame of the j'th derivative of the latent dynamics when we sample the latent dynamics
        coefficients using the posterior distribution for the specified parameter values.
        """

        """
        Time integrates the latent dynamics using the coefficients specified in coefs when we
        start from the initial condition(s) in IC.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        coefs: Either a one or two dimensional numpy.ndarray or torch.Tensor objects. If it has 
        one dimension, then coefs represents a is a flattened copy of hstack[-K, -C, b], where 
        K, C, and b are the optimal coefficients for a specific combination of parameter values. 
        If coefs has two dimensions, then it should have shape (n_param, n_coef) and it's i'th row 
        should represent the optimal set of coefficients when we use the i'th combination of 
        parameter values. In this case, we inductively call simulate on each row of coefs. 

        IC: A list of n_IC numpy.ndarray or torch.Tensor objects. These objects should have either 
        two or three dimensions. If coefs has one dimension, then each element of IC should have 
        shape (n, n_z) where n represents the number of initial conditions we want to simulate 
        forward in time and n_z is the latent dimension. If you want to simulate a single IC, then 
        n == 1. If coefs has two dimensions (specifically if coefs.shape = (n_param, n_coef)), then 
        each element of IC should have shape (n_param, n, n_z). In this case, IC[d][i, j, :] 
        represents the j'th initial condition for the d'th derivative of the latent state when 
        we use the i'th combination of parameter values.

        times: A 1d numpy ndarray object whose i'th entry holds the value of the i'th time value 
        where we want to compute the latent solution. The elements of this array should be in 
        ascending order. 


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------        
        
        A list of n_IC numpy ndarrays representing the simulated displacements and velocities. If 
        coefs has two dimensions and shape (n_param, n_coefs), then each array should have shape 
        (n_param, n_t, n_z). The i,j,k element of the arrays should hold the k'th component of 
        the j'th time steps of the simulated displacement and velocity, respectively, when we use 
        the i'th set of coefficients (coefs[i, :]) to simulate the latent dynamics.
        
        If coefs has one dimension, the then each array will have two dimensions and shape 
        (n_t, n_z). The i, j element of the returned arrays should house the j'th component of the
        displacement and velocity at the i'th time step, respectively.
        """

        raise RuntimeError('Abstract function LatentDynamics.simulate!');
    


    def export(self) -> dict:
        param_dict = {'n_z'     : self.n_z, 
                      'n_coefs' : self.n_coefs, 
                      'n_IC'    : self.n_IC};
        return param_dict;



    # SINDy does not need to load parameters.
    # Other latent dynamics might need to.
    def load(self, dict_ : dict) -> None:
        assert(self.n_z         == dict_['n_z']);
        assert(self.n_coefs     == dict_['n_coefs']);
        assert(self.n_IC        == dict_['n_IC']);
        return;
    