# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

# Add the main directory to the search path.
import  os;
import  sys;
src_Path        : str   = os.path.dirname(os.path.dirname(__file__));
util_Path       : str   = os.path.join(src_Path, "Utilities");
sys.path.append(src_Path);
sys.path.append(util_Path);

import  numpy;
import  torch;

from    LatentDynamics      import  LatentDynamics;
from    InputParser         import  InputParser;
from    FiniteDifference    import  Derivative1_Order4, Derivative2_Order4;
from    Solvers             import  RK4;



# -------------------------------------------------------------------------------------------------
# DampedSpring class
# -------------------------------------------------------------------------------------------------

class DampedSpring(LatentDynamics):
    def __init__(self, 
                 dim        : int, 
                 nt         : int, 
                 config     : dict) -> None:
        r"""
        Initializes a DampedSpring object. This is a subclass of the LatentDynamics class which 
        implements the following latent dynamics
            z''(t) = -K z(t) - C z'(t) + b
        Here, z is the latent state. K \in \mathbb{R}^{n x n} represents a generalized spring 
        matrix, C represents a damping matrix, and b is an offset/constant forcing function. 
        In this expression, K, C, and b are the model's coefficients. There is a separate set of
        coefficients for each combination of parameter values. 
            

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        dim: The number of dimensions in the latent space, where the latent dynamics takes place.

        nt: The number of time steps we want to generate when solving (numerically) the latent 
        space dynamics.

        config: A dictionary housing the settings we need to set up a DampedSpring object. 
        Specifically, this dictionary should have a key called "spring" whose corresponding value 
        is another dictionary with the following key:
            - coef_norm_order: A string specifying which norm we want to use when computing
            the coefficient loss.
        """

        # Run the base class initializer. The only thing this does is set the dim and nt 
        # attributes.;
        super().__init__(dim, nt);

        # store the dimension of the latent dynamics.
        self.dim    : int   = dim;

        # Now, set up an Input parser to read in the coefficient norm order.
        assert('spring' in config);
        spring_parser           = InputParser(config['spring'], name = 'spring_input');
        self.coef_norm_order    = spring_parser.getInput(['coef_norm_order'], fallback = 1);

        # Set up the loss function for the latent dynamics.
        self.LD_LossFunction = torch.nn.MSELoss();

        # All done!
        return;
    


    def calibrate(self, 
                  Z             : torch.Tensor,
                  dt            : float, 
                  numpy         : bool = False) -> tuple[(numpy.ndarray | torch.Tensor), torch.Tensor, torch.Tensor]:
        r"""
        For each combination of parameter values, this function computes the optimal K, C, and b 
        coefficients in the sequence of latent states for that combination of parameter values.
        
        Specifically, let us consider the case when Z has two axes (the case when it has three is 
        identical, just with different coefficients for each instance of the leading dimension of 
        Z). In this case, we assume the i'th row of Z holds the latent state t_0 + i*dt. We use 
        We assume that the latent state is governed by an ODE of the form
            z''(t) = -K z(t) - C z'(t) + b
        We find K, C, and b corresponding to the dynamical system that best agrees with the 
        snapshots in the rows of Z (the K, C, and b which minimize the mean square difference 
        between the left and right hand side of this equation across the snapshots in the rows 
        of Z).


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Z: A 2d or 3d tensor. If Z is a 2d tensor, then it has shape (Nt, Nz), where Nt specifies 
        the length of the sequence of latent states and Nz is the dimension of the latent space. In 
        this case, the i,j entry of Z holds the j'th component of the latent state at the time 
        t_0 + i*dt. If it is a 3d tensor, then it has shape (Np, Nt, Nz). In this case, we assume 
        there are Np different combinations of parameter values. The i, j, k entry of Z in this 
        case holds the k'th component of the latent encoding at time t_0 + j*dt when we use the 
        i'th combination of parameter values. 

        dt: The time step between time steps. See the description of the "Z" argument. 

        numpy: A boolean. If True, we return the coefficient matrix as a numpy.ndarray object. If 
        False, we return it as a torch.Tensor object.
        

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        We return three variables. 
        
        The first holds the coefficients. It is a matrix of shape (n_train, dim*(2*dim + 1)), 
        where n_train is the number of parameter combinations in the training set and dim is the 
        dimension of the latent space. The i'th row holds the flattened version of 
        hstack(K, C, b), where K, C, and b are the optimal coefficients for the time series 
        corresponding to the i'th combination of parameter values.

        The second holds the total Latent Dynamics loss. It is a single element tensor whose lone 
        entry holds the sum of the Latent Dynamics (left minus right hand side of the ODE) across 
        the set of combinations of parameters in the training set. 

        The third is a single element tensor whose lone element holds the sum of the L1 norms of 
        the coefficients across the set of combinations of parameters in the training set.
        """

        # -----------------------------------------------------------------------------------------
        # If Z has three dimensions, loop over all train cases.
        if (Z.dim() == 3):
            # Fetch the number of training cases.
            n_train : int = Z.size(0);

            # Prepare an array to house the flattened coefficient matrices for each combination of
            # parameter values.
            if (numpy):
                coefs = numpy.zeros([n_train, self.dim*(2*self.dim + 1)]);
            else:
                coefs = torch.Tensor([n_train, self.dim*(2*self.dim + 1)]);

            # Initialize the losses.
            Loss_LD     : torch.Tensor  = torch.tensor(0, dtype = torch.float32);
            Loss_Coef   : torch.Tensor  = torch.tensor(0, dtype = torch.float32);

            # Cycle through the combinations of parameter values.
            for i in range(n_train):
                """"
                Get the optimal K, C, and b coefficients for the i'th combination of parameter 
                values. 
                
                Remember that Z is 3d tensor of shape (Np, Nt, Nz) whose (i, j, k) entry holds the 
                k'th component of the j'th frame of the latent trajectory for the i'th combination 
                of parameter values. 
                """
                result : tuple = self.calibrate(Z[i], dt, numpy);

                # Package everything from this combination of training values.
                coefs[i]    = result[0];
                Loss_LD    += result[1];
                Loss_Coef  += result[2];
            
            # All done!
            return coefs, Loss_LD, Loss_Coef;
            


        # -----------------------------------------------------------------------------------------
        # evaluate for one training case.
        assert(Z.dim() == 2)

        # First, compute the time derivatives. 
        dZ_dt   : torch.Tensor  = Derivative1_Order4(X = Z, h = dt);
        d2Z_dt2 : torch.Tensor  = Derivative2_Order4(X = Z, h = dt);

        # Concatenate Z, dZ_dt and a column of 1's. We will solve for the matrix, E, which gives 
        # the best fit for the system d2Z_dt2 = cat[Z, dZ_dt, 1] E. This matrix has the form 
        # E^T = [-K, -C, b]. Thus, we can extract K, C, and b from W.
        W       : torch.Tensor  = torch.cat([Z, dZ_dt, torch.ones(Z.shape[0], 1)], dim = 1);
        
        # For each j, solve the least squares problem 
        #   min{ || d2Z_dt2[:, j] - W E(j)|| : E(j) \in \mathbb{R}^(dim*(2*dim + 1)) }
        # We store the resulting solutions in a matrix, coefs, whose j'th column holds the 
        # results for the j'th column of dZdt. Thus, coefs is a 2d tensor with shape 
        # (dim(2*dim + 1), Nz).
        coefs   : torch.Tensor  = torch.linalg.lstsq(W, d2Z_dt2).solution;

        # Compute the losses
        Loss_LD     = self.LD_LossFunction(d2Z_dt2, W @ coefs);
        Loss_Coef   = torch.norm(coefs, self.coef_norm_order);

        # All done. Prepare coefs and the losses to return. Note that we flatten the coefficient 
        # matrix.
        # Note: output of lstsq is not contiguous in memory.
        coefs = coefs.detach().flatten();
        if (numpy):
            coefs = coefs.numpy();

        return coefs, Loss_LD, Loss_Coef;
    


    def simulate(   self,
                    coefs   : numpy.ndarray, 
                    IC      : numpy.ndarray,
                    times   : numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        Time integrates the latent dynamics when it uses the coefficients specified in coefs and 
        starts from the (single) initial condition in z0.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        coefs: A one dimensional numpy.ndarray object representing the flattened copy of 
        hstack[-K, -C, b]. We extract K, C, and b from coefs.

        IC: A 2d numpy ndarray of shape 2 x dim. The two rows of this array represent the initial
        displacement and position of the latent dynamics, respectively. Thus, the i'th component 
        of these arrays should hold the i'th component of the latent dynamics initial displacement 
        and velocity, respectively.
        
        times: A 1d numpy ndarray object whose i'th entry holds the value of the i'th time value 
        where we want to compute the latent solution. The elements of this array should be in 
        ascending order. 


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------        
        
        A 3d numpy ndarray of shape (2, nt, dim) where nt is the number of time steps (size of 
        times) and dim is the latent space dimension (self.dim). The 0,i,j and 1,i,j elements of 
        this array hold the j'th components of the displacement and velocity at the i'th time step, 
        respectively.
        """

        # Run checks.
        assert(len(IC.shape)        == 2);
        assert(IC.shape[0]          == 2);
        assert(IC.shape[1]          == self.dim);
        assert(len(times.shape)     == 1);

        # First, we need to extract -K, -C, and b from coefs. We know that coefs is the least 
        # squares solution to d2Z_dt2 = hstack[Z, dZdt, 1] E^T. Thus, we expect that.
        # E = [-K, -C, b]. 
        E   : numpy.ndarray = coefs.reshape([self.dim, 2*self.dim + 1]).T;

        # Extract K, C, and b.
        K   : numpy.ndarray = -E[:, 0:self.dim];
        C   : numpy.ndarray = -E[:, self.dim:(2*self.dim)];
        b   : numpy.ndarray = E[:, 2*self.dim:(2*self.dim + 1)];

        # Set up a lambda function to approximate (d^2/dt^2)z(t) \approx -K z(t) - C (d/dt)z(t) + b
        f    = lambda t, z, dz_dt : -numpy.matmul(K, z) - numpy.matmul(C, dz_dt) + b;

        # Solve the ODE forward in time.
        Z, dZ_dt = RK4(f = f, y0 = IC[0], Dy0 = IC[1], times = times);

        # All done!
        return Z, dZ_dt;
    


    def export(self) -> dict:
        """
        This function packages self's contents into a dictionary which it then returns. We can use 
        this dictionary to create a new DampedSpring object which has the same internal state as 
        self. 
        
        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        None!


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        A dictionary with one key: coef_norm_order. It's value specifies which norm we want to use 
        when computing the coefficient loss. 
        """

        param_dict                      = super().export();
        param_dict['coef_norm_order']   = self.coef_norm_order;

        return param_dict;
        