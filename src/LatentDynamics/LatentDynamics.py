# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  numpy;
import  torch;



# -------------------------------------------------------------------------------------------------
# LatentDynamics base class
# -------------------------------------------------------------------------------------------------

class LatentDynamics:
    # Class variables
    dim     : int           = -1        # Dimensionality of the latent space
    nt      : int           = -1        # Number of time steps when solving the latent dynamics
    ncoefs  : int           = -1        # Number of coefficients in the latent space dynamics

    # TODO(kevin): do we want to store coefficients as an instance variable?
    coefs   : torch.Tensor  = torch.Tensor([])



    def __init__(self, dim_ : int, nt_ : int) -> None:
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

        nt_: The number of time steps we want to generate when solving (numerically) the latent 
        space dynamics.
        """

        # Set class variables.
        self.dim    : int   = dim_
        self.nt     : int   = nt_

        # There must be at least one latent dimension and there must be at least 1 time step.
        assert(self.dim > 0)
        assert(self.nt  > 0)

        # All done!
        return
    


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
        Z has two dimensions, it has shape (Nt, Nz), where Nt specifies the number of time steps in 
        each sequence of latent states and Nz is the dimension of the latent space. In this case, 
        the i,j entry of Z holds the j'th component of the latent state  at the time t_0 + i*dt. If 
        it is a 3d tensor, then it has shape (Np, Nt, Nz). In this case, we assume there at Np 
        different combinations of parameter values. The i, j, k entry of Z in this case holds the 
        k'th component of the latent encoding at time t_0 + j*dt when we use the i'th combination 
        of parameter values. 

        dt: The time step between time steps. See the description of the "Z" argument. 

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
     
        A torch.Tensor holding the optimal coefficients for the latent space dynamics given the 
        data stored in Z. If Z is 2d, then the returned tensor will only contain one set of 
        coefficients. If Z is 3d, with a leading dimension size of Np (number of combinations of 
        parameter values) then we will return an array/tensor with a leading dimension of size Np 
        whose i'th entry holds the coefficients for the sequence of latent states stored in 
        Z[:, ...].
        """

        raise RuntimeError('Abstract function LatentDynamics.calibrate!')
    


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
        
        Specifically, the i'th element is a 2d array of shape (nt, dim), where nt is the number 
        of time steps (size of times) and dim is the latent space dimension (self.dim). Thus, 
        the j,k element of this matrix holds the k'th component of the i'th time derivative of the
        latent solution at the time stored in the j'th element of times. 
        """

        raise RuntimeError('Abstract function LatentDynamics.simulate!')
    


    def sample( self, 
                coefs_Sample    : numpy.ndarray, 
                IC_Samples      : list[list[numpy.ndarray]], 
                times           : numpy.ndarray) -> numpy.ndarray:
        """
        Simulate's the latent dynamics for a set of coefficients/initial conditions.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        coefs_Sample: A numpy.ndarray object whose leading dimension has size ns (the number of 
        sets of coefficients/initial conditions/simulations we run).

        IC_Samples: A list of lists list of numpy arrays. The i'th element of this list should be 
        a list of length n_IC whose j'th element is a 1d numpy ndarray object of shape dim, where 
        dim is the dimension of the latent space (the space where the dynamics takes place). Here, 
        ns is the number of samples we want to process and n_IC is the number of initial conditions
        we need to specify to define the initial state of the latent dynamics.

        times: A 1d numpy ndarray object whose i'th entry holds the value of the i'th time value 
        where we want to compute each latent solution. The elements of this array should be in 
        ascending order. We use the same array for each set of coefficients.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        A 4d numpy ndarray object of shape (n_IC, ns, nt, dim), where n_IC = the number of 
        derivatives of the initial state we need to specify in the initial conditions, ns = the 
        number of samples (the length of IC_Samples), nt = the number of time steps (size of times) 
        and dim is the dimension of the latent space. The i, j, k, l element of this array holds 
        the l'th component of the solution of the latent dynamics at the j'th time step (j'th 
        element of times) when we use the i'th set of coefficients/initial conditions. 
        """

        # There needs to be as many initial conditions as sets of coefficients.
        assert(len(IC_Samples)          == coefs_Sample.shape[0]);

        # Fetch ns, n_IC.
        ns      : int   = len(IC_Samples)
        n_IC    : int   = IC_Samples.shape[1];

        # Cycle through the set of coefficients
        for i in range(ns):
            # Simulate the latent dynamics when we use the i'th set of coefficients + ICs
            Z_i : numpy.ndarray = self.simulate(coefs  = coefs_Sample[i], 
                                                IC     = IC_Samples[i], 
                                                times  = times);

            # Append a leading dimension of size 1.
            Z_i = Z_i.reshape(n_IC, 1, Z_i.shape[0], Z_i.shape[1]);

            # Append the latest trajectory onto the Z_simulated array.
            if (i == 0):
                Z_simulated = Z_i;
            else:
                Z_simulated = numpy.concatenate((Z_simulated, Z_i), axis = 1);

        # All done!
        return Z_simulated



    def export(self) -> dict:
        param_dict = {'dim': self.dim, 'ncoefs': self.ncoefs}
        return param_dict



    # SINDy does not need to load parameters.
    # Other latent dynamics might need to.
    def load(self, dict_ : dict) -> None:
        assert(self.dim == dict_['dim'])
        assert(self.ncoefs == dict_['ncoefs'])
        return
    