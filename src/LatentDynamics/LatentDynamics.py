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
        Time integrates the latent dynamics when it uses the coefficients specified in coefs and 
        starts from the (single) initial condition in z0. The user must implement this method.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        coefs: A one dimensional numpy.ndarray object holding the coefficients we want to use 
        to solve the latent dynamics forward in time. 
        
        IC: A list of n_IC numpy.ndarray objects, each of shape dim. Here, n_IC is the number of 
        initial conditions we need to specify to define the initial state of the latent dynamics. 
        Likewise, dim the the dimension of the latent space (the space where the dynamics take 
        place). The j'th element of the i'th element of this list should hold the j'th component of 
        the initial condition for the i'th derivative of the initial condition. 

        times: A 1d numpy ndarray object whose i'th entry holds the value of the i'th time value 
        where we want to compute the latent solution. The elements of this array should be in 
        ascending order.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------        
        
        A list of 2d numpy.ndarray object holding the solution to the latent dynamics and its 
        time derivatives at the time values specified in times when we use the coefficients in 
        coefs to characterize the latent dynamics model. 
        
        Specifically, the i'th element is a 2d array of shape (n_t, dim), where n_t is the number 
        of time steps (size of times) and dim is the latent space dimension (self.dim). Thus, 
        the j,k element of this matrix holds the k'th component of the i'th time derivative of the
        latent solution at the time stored in the j'th element of times. 
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
    