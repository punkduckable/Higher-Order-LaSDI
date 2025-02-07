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
    dim     : int           = -1;       # Dimensionality of the latent space
    n_t      : int           = -1;      # Number of time steps when solving the latent dynamics
    n_coefs : int           = -1;       # Number of coefficients in the latent space dynamics
    n_IC    : int           = -1;       # Number of initial conditions to define the initial latent state.

    # TODO(kevin): do we want to store coefficients as an instance variable?
    coefs   : torch.Tensor  = torch.Tensor([]);



    def __init__(self, dim_ : int, n_t_ : int) -> None:
        """
        Initializes a LatentDynamics object. Each LatentDynamics object needs to have a 
        dimensionality (dim), a number of time steps, a model for the latent space dynamics, and 
        set of coefficients for that model. The model should describe a set of ODEs in 
        \mathbb{R}^{dim}. These ODEs should contain a set of unknown coefficients. We learn those 
        coefficients using the calibrate function. Once we have learned the coefficients, we can 
        solve the corresponding set of ODEs forward in time using the simulate function.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        dim_: The number of dimensions in the latent space, where the latent dynamics takes place.

        n_t_: The number of time steps we want to generate when solving (numerically) the latent 
        space dynamics.
        """

        # Set class variables.
        self.dim    : int   = dim_;
        self.n_t    : int   = n_t_;

        # There must be at least one latent dimension and there must be at least 1 time step.
        assert(self.dim > 0);
        assert(self.n_t  > 0);

        # All done!
        return;
    


    def calibrate(  self, 
                    Latent_States : list[torch.Tensor], 
                    dt            : int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The user must implement this class on any latent dynamics sub-class. Each latent dynamics 
        object should implement a parameterized model for the dynamics in the latent space. A 
        Latent_Dynamics object should pair each combination of parameter values with a set of 
        coefficients in the latent space. Using those parameters, we compute loss functions (one 
        characterizing how well the left and right hand side of the latent dynamics match, another
        specifies the norm of the coefficient matrix). 

        This function computes the optimal coefficients and the losses, which it returns.

        Specifically, this function should take in a sequence (or sequences) of latent states, a
        time step, dt, which specifies the time step between successive terms in the sequence(s) of
        latent states, and some optional booleans which control what information we return. 


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_States: A list of 2d or 3d tensor holding the latent position, velocity, etc. Here, 
        we will consider the case when len(Latent_Sates) = 1 and will let Z = Latent_States[0]. If 
        Z has two dimensions, it has shape (n_t, n_z), where n_t specifies the number of time steps in 
        each sequence of latent states and n_Z is the dimension of the latent space. In this case, 
        the i,j entry of Z holds the j'th component of the latent state  at the time t_0 + i*dt. If 
        it is a 3d tensor, then it has shape (n_param, n_t, n_Z). In this case, we assume there at 
        n_param different combinations of parameter values. The i, j, k entry of Z in this case holds 
        the k'th component of the latent encoding at time t_0 + j*dt when we use the i'th combination 
        of parameter values. 

        dt: The time step between time steps. See the description of the "Z" argument. 

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
     
        A torch.Tensor holding the optimal coefficients for the latent space dynamics given the 
        data stored in Z. If Z is 2d, then the returned tensor will be a 1d tensor of shape 
        (self.n_coefs). If Z is 3d, with a leading dimension size of Np (number of combinations of 
        parameter values) then we will return a 2d tensor of shape (Np, self.n_coefs) whose i, j 
        entry holds the value of the j'th coefficient when we use the i'th combination of parameter
        values. 
        """

        raise RuntimeError('Abstract function LatentDynamics.calibrate!');
    


    def simulate(self, 
                 coefs  : numpy.ndarray, 
                 IC     : list[numpy.ndarray], 
                 times  : numpy.ndarray) -> list[numpy.ndarray]:
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
        shape (n, dim) where n represents the number of initial conditions we want to simulate 
        forward in time and dim is the latent dimension. If you want to simulate a single IC, then 
        n == 1. If coefs has two dimensions (specifically if coefs.shape = (n_param, n_coef)), then 
        each element of IC should have shape (n_param, n, dim). In this case, IC[d][i, j, :] 
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
        (n_param, n_t, dim). The i,j,k element of the arrays should hold the k'th component of 
        the j'th time steps of the simulated displacement and velocity, respectively, when we use 
        the i'th set of coefficients (coefs[i, :]) to simulate the latent dynamics.
        
        If coefs has one dimension, the then each array will have two dimensions and shape 
        (n_t, dim). The i, j element of the returned arrays should house the j'th component of the
        displacement and velocity at the i'th time step, respectively.
        """

        raise RuntimeError('Abstract function LatentDynamics.simulate!');
    


    def export(self) -> dict:
        param_dict = {'dim'     : self.dim, 
                      'n_coefs' : self.n_coefs, 
                      'n_IC'    : self.n_IC};
        return param_dict;



    # SINDy does not need to load parameters.
    # Other latent dynamics might need to.
    def load(self, dict_ : dict) -> None:
        assert(self.dim         == dict_['dim']);
        assert(self.n_coefs     == dict_['n_coefs']);
        assert(self.n_IC        == dict_['n_IC']);
        return;
    