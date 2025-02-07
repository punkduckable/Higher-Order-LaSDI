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

import  logging;

import  numpy;
import  torch;

from    LatentDynamics      import  LatentDynamics;
from    FiniteDifference    import  Derivative1_Order4, Derivative2_Order4;
from    Solvers             import  RK4;


# Setup Logger.
LOGGER : logging.Logger = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# DampedSpring class
# -------------------------------------------------------------------------------------------------

class DampedSpring(LatentDynamics):
    def __init__(self, 
                 dim        : int, 
                 n_t        : int, 
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

        n_t: The number of time steps we want to generate when solving (numerically) the latent 
        space dynamics.

        config: A dictionary housing the settings we need to set up a DampedSpring object. 
        Specifically, this dictionary should have a key called "spring" whose corresponding value 
        is another dictionary with the following key:
            - coef_norm_order: A string specifying which norm we want to use when computing
            the coefficient loss.
        """

        # Run the base class initializer. The only thing this does is set the dim and n_t 
        # attributes.;
        super().__init__(dim, n_t);
        LOGGER.info("Initializing a DampedSpring object with dim = %d, n_t = %d" % (self.dim, self.n_t));
        
        # Set n_coefs and n_IC.
        # Because K and C are dim x dim matrices, and b is in \mathbb{R}^dim, there are 
        # dim*(2*dim + 1) coefficients in the latent dynamics.
        self.n_IC       : int   = 2;
        self.n_coefs    : int   = dim*(2*dim + 1);

        # Now, read the coefficient norm order.
        assert('spring' in config);
        self.coef_norm_order    = config['spring']['coef_norm_order'];

        # Set up the loss function for the latent dynamics.
        self.LD_LossFunction = torch.nn.MSELoss();

        # All done!
        return;
    


    def calibrate(self, 
                  Latent_States : list[torch.Tensor],
                  dt            : float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        Latent_States: A two element list whose elements hold the latent positions and velocities, 
        respectively. Each list element is a 2d or 3d tensor. If they are 2d tensors, then each 
        has shape (Nt, Nz), where Nt specifies the length of the sequence of latent states and Nz 
        is the dimension of the latent space. In this case, the i,j entry of Latent_States[0], 
        Latent_States[1] holds the j'th component of the latent state and the latent velocity, 
        respectively, at the i'th time. If they are is a 3d tensors, then each has shape (Np, Nt, 
        Nz). In this case, we assume there are Np different combinations of parameter values. The 
        i, j, k entry of Latent_States[0], Latent_States[1] hold the k'th component of the latent 
        encoding and the latent velocity at the j'th time when we use the i'th combination of 
        parameter values. 

        dt: The time step between time steps. See the description of the "Latent_States" argument. 
        

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        We return three variables. 
        
        The first holds the coefficients. It is a torch.Tensor of shape (n_train, dim*(2*dim + 1)), 
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

        # Run checks.
        assert(len(Latent_States)       == 2);
        assert(Latent_States[0].shape   == Latent_States[1].shape);

        # Extract the latent states and their velocities.
        Z_X     : torch.Tensor  = Latent_States[0];
        Z_V     : torch.Tensor  = Latent_States[1];
        

        # -----------------------------------------------------------------------------------------
        # If Z has three dimensions, loop over all train cases.
        if (len(Z_X.shape) == 3):
            # Fetch the number of training cases.
            n_train : int = Z_X.shape[0];
            LOGGER.debug("Finding the optimal coefficients for %d parameter values" % n_train);

            # Prepare an array to house the flattened coefficient matrices for each combination of
            # parameter values.
            coefs = torch.empty([n_train, self.dim*(2*self.dim + 1)], dtype = torch.float32);

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
                result : tuple[torch.Tensor] = self.calibrate([Z_X[i], Z_V[i]], dt);

                # Package everything from this combination of training values.
                coefs[i, :] = result[0];
                Loss_LD    += result[1];
                Loss_Coef  += result[2];
            
            # All done!
            return coefs, Loss_LD, Loss_Coef;
            


        # -----------------------------------------------------------------------------------------
        # evaluate for one training case.
        assert(len(Z_X.shape) == 2);
        
        # First, compute the second time derivative of Z_X. This should also be the first time 
        # derivative of Z_V. We average the two so that the final loss depends on both.
        d2Z_dt2_from_Z_X    : torch.Tensor  = Derivative2_Order4(X = Z_X,   h = dt);
        d2Z_dt2_from_Z_V    : torch.Tensor  = Derivative1_Order4(X = Z_V,   h = dt);
        d2Z_dt2             : torch.Tensor  = 0.5*d2Z_dt2_from_Z_X + 0.5*d2Z_dt2_from_Z_V;

        # Concatenate Z_X, Z_V and a column of 1's. We will solve for the matrix, E, which gives 
        # the best fit for the system d2Z_dt2 = cat[Z_X, Z_V, 1] E. This matrix has the form 
        # E^T = [-K, -C, b]. Thus, we can extract K, C, and b from W.
        W       : torch.Tensor  = torch.cat([Z_X, Z_V, torch.ones((Z_X.shape[0], 1))], dim = 1);

        # For each j, solve the least squares problem 
        #   min{ || d2Z_dt2[:, j] - W E(j)|| : E(j) \in \mathbb{R}^(dim*(2*dim + 1)) }
        # We store the resulting solutions in a matrix, coefs, whose j'th column holds the 
        # results for the j'th column of Z_V. Thus, coefs is a 2d tensor with shape 
        # (2*dim + 1, dim).
        coefs   : torch.Tensor  = torch.linalg.lstsq(W, d2Z_dt2).solution;

        # Compute the losses
        Loss_LD     = self.LD_LossFunction(d2Z_dt2, torch.matmul(W, coefs));
        Loss_Coef   = torch.norm(coefs, self.coef_norm_order);

        if(False):
            # Extract K, C, and b from coefs.
            E   : torch.Tensor  = coefs.T;
            K   : torch.Tensor  = -E[:, 0:self.dim];
            C   : torch.Tensor  = -E[:, self.dim:(2*self.dim)];
            b   : torch.Tensor  = E[:, 2*self.dim:(2*self.dim + 1)];
            
            # Compute the RHS of the diff eq using coefs and the matrices we found.
            RHS_coefs           = torch.matmul(W, coefs);
            RHS_Manual          = torch.matmul(torch.ones((Z_X.shape[0], 1)), b.T) - torch.matmul(Z_V, C.T) - torch.matmul(Z_X, K.T);

            # Select a random row to sample.
            import random;
            row : int           = random.randint(a = 0, b = Z_X.shape[0]);

            print("Row %d of RHS using coefs:                   %s" % (row, str(RHS_coefs[row, :])));
            print("Row %d of RHS using K, C, and b:             %s" % (row, str(RHS_Manual[row, :])));
            print("Max diff between RHS with coefs and K/C/b:   %f" % torch.max(torch.abs(RHS_coefs - RHS_Manual)));

        # Prepare coefs and the losses to return. Note: we flatten the coefficient matrix.
        # Note: output of lstsq is not contiguous in memory.
        coefs   : torch.Tensor  = coefs.detach().flatten();
        return coefs, Loss_LD, Loss_Coef;
    


    def simulate(   self,
                    coefs   : numpy.ndarray         | torch.Tensor, 
                    IC      : list[numpy.ndarray]   | list[torch.Tensor],
                    times   : numpy.ndarray) -> list[numpy.ndarray]  | list[torch.Tensor]:
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

        IC: A list of two numpy.ndarray or torch.Tensor objects. These objects should have either 
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
        
        A list of two numpy ndarrays representing the simulated displacements and velocities. If 
        coefs has two dimensions and shape (n_param, n_coefs), then each array should have shape 
        (n_param, n_t, dim). The i,j,k element of the two arrays should hold the k'th component of 
        the j'th time steps of the simulated displacement and velocity, respectively, when we use 
        the i'th set of coefficients (coefs[i, :]) to simulate the latent dynamics.
        
        If coefs has one dimension, the then each array will have two dimensions and shape 
        (n_t, dim). The i, j element of the returned arrays should house the j'th component of the
        displacement and velocity at the i'th time step, respectively.
        """

        # Run checks.
        assert(len(IC)              == 2);
        assert((len(coefs.shape)    == 1    and len(IC[0].shape)    == 2)    or  (len(coefs.shape)    == 2  and len(IC[0].shape)    == 3));
        assert(IC[0].shape[-1]      == self.dim);
        assert(IC[0].shape          == IC[1].shape);
        assert(len(times.shape)     == 1);
        assert(type(coefs)          == type(IC[0]));
        assert(type(coefs)          == type(IC[1]));


        # The way this function works depends on if there is one set of coefficients or an entire
        # batch of them. 
        if(len(coefs.shape) == 2):
            # In this case, coefs.shape = (n_param, n_coefs) and each element of IC has shape 
            # (n_param, n, dim). First, let's extract n_parm, n_t, and n.
            n_param     : int       =   coefs.shape[0];
            assert(IC[0].shape[0]   ==  n_param);
            n_t         : int       =   times.size;
            n           : int       =   IC[0].shape[1];
            LOGGER.debug("Simulating with %d parameter combinations" % n_param);

            # Set up arrays to hold the simulated positions and velocities.
            if(isinstance(coefs, numpy.ndarray)):
                D   : numpy.ndarray     = numpy.empty((n_param, n_t, n, self.dim), dtype = numpy.float32);
                V   : numpy.ndarray     = numpy.empty((n_param, n_t, n, self.dim), dtype = numpy.float32);
            elif(isinstance(coefs, torch.Tensor)):
                D   : torch.Tensor      = torch.empty((n_param, n_t, n, self.dim), dtype = torch.float32);
                V   : torch.Tensor      = torch.empty((n_param, n_t, n, self.dim), dtype = torch.float32);             

            # Now, cycle through the parameter combinations
            for i in range(n_param):
                # Extract the i'th combinations of coefficients and initial conditions.
                ith_coefs   : numpy.ndarray | torch.Tensor              = coefs[i, :];
                ith_IC      : list[numpy.ndarray] | list[torch.Tensor]  = [IC[0][i, :, :], IC[1][i, :, :]];

                # Call this function using them.
                ith_Results : list[numpy.ndarray] | list[torch.Tensor]  = self.simulate(
                                                                                    coefs   = ith_coefs, 
                                                                                    IC      = ith_IC, 
                                                                                    times   = times);

                # Add these results to D, V.
                D[i, :, :, :]  = ith_Results[0];
                V[i, :, :, :]  = ith_Results[1];

            # All done.
            return [D, V];
    

        # If we get here, then coefs has one dimension. In this case, each element of IC should 
        # have shape (n, dim). 
        D0  : numpy.ndarray | torch.Tensor  = IC[0]; 
        V0  : numpy.ndarray | torch.Tensor  = IC[1];

        # First, we need to extract -K, -C, and b from coefs. We know that coefs is the least 
        # squares solution to d2Z_dt2 = hstack[Z, dZdt, 1] E^T. Thus, we expect that.
        # E = [-K, -C, b]. 
        E   : numpy.ndarray | torch.Tensor = coefs.reshape([2*self.dim + 1, self.dim]).T;

        # Extract K, C, and b. Note that we need to reshape b to have shape (1, dim) to enable
        # broadcasting.
        K   : numpy.ndarray | torch.Tensor = -E[:, 0:self.dim];
        C   : numpy.ndarray | torch.Tensor = -E[:, self.dim:(2*self.dim)];
        b   : numpy.ndarray | torch.Tensor = E[:, 2*self.dim].reshape(1, -1);

        # Set up a lambda function to approximate (d^2/dt^2)z(t) \approx -K z(t) - C (d/dt)z(t) + b.
        # In this case, we expect dz_dt and z to have shape (n, dim). Thus, matmul(z, K.T) will have 
        # shape (n, dim). The i'th row of this should hold the z portion of the rhs of the latent
        # dynamics for the i'th IC. Similar results hold for dot(dz_dt, C.T). The final result 
        # should have shape (n, dim). The i'th row should hold the rhs of the latent dynamics 
        # for the i'th IC.
        if(isinstance(coefs, numpy.ndarray)):
            f   = lambda t, z, dz_dt: b - numpy.matmul(dz_dt, C.T)  - numpy.matmul(z, K.T);
        if(isinstance(coefs, torch.Tensor)):
            f   = lambda t, z, dz_dt: b - torch.matmul(dz_dt, C.T)  - torch.matmul(z, K.T);

        # Solve the ODE forward in time. D and V should have shape (n_t, n, dim).
        D, V = RK4(f = f, y0 = D0, Dy0 = V0, times = times);

        # All done!
        return [D, V];
    


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
        