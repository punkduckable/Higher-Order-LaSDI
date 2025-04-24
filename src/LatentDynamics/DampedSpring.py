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
from    FiniteDifference    import  Derivative1_Order4, Derivative2_Order4, Derivative1_Order2_NonUniform, Derivative2_Order2_NonUniform;
from    SecondOrderSolvers  import  RK4;


# Setup Logger.
LOGGER : logging.Logger = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# DampedSpring class
# -------------------------------------------------------------------------------------------------

class DampedSpring(LatentDynamics):
    def __init__(   self, 
                    n_z             :   int, 
                    coef_norm_order :   str | float, 
                    Uniform_t_Grid  :   bool) -> None:
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

        # Run the base class initializer. The only thing this does is set the n_z and n_t 
        # attributes.;
        super().__init__(   n_z             = n_z,
                            coef_norm_order = coef_norm_order,
                            Uniform_t_Grid  = Uniform_t_Grid);
        LOGGER.info("Initializing a SINDY object with n_z = %d, coef_norm_order = %s, Uniform_t_Grid = %s" % (  self.n_z, 
                                                                                                                str(self.coef_norm_order), 
                                                                                                                str(self.Uniform_t_Grid)));        
        
        # Set n_coefs and n_IC.
        # Because K and C are n_z x n_z matrices, and b is in \mathbb{R}^n_z, there are 
        # n_z*(2*n_z + 1) coefficients in the latent dynamics.
        self.n_IC       : int   = 2;
        self.n_coefs    : int   = n_z*(2*n_z + 1);

        # Setup the loss function.
        self.LD_LossFunction = torch.nn.MSELoss();
        return;
    


    def calibrate(self, 
                  Latent_States : list[torch.Tensor],
                  t_Grid        : list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        Latent_States: An n_param (number of parameter combinations we want to calibrate) element
        list. The i'th list element should be an 2 element list whose j'th element is a 2d numpy 
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

        # Run checks.
        assert(isinstance(t_Grid, list));
        assert(isinstance(Latent_States, list));
        assert(len(Latent_States)   == len(t_Grid));

        n_param : int   = len(t_Grid);
        n_IC    : int   = 2;
        n_z     : int   = self.n_z;
        for i in range(n_param):
            assert(isinstance(Latent_States[i], list));
            assert(len(Latent_States[i]) == n_IC);

            for j in range(n_IC):
                assert(isinstance(Latent_States[i][j], torch.Tensor));
                assert(len(Latent_States[i][j].shape)   == 2);
                assert(Latent_States[i][j].shape[-1]    == n_z);


        # -----------------------------------------------------------------------------------------
        # If there are multiple combinations of parameter values, loop through them.
        
        if (n_param > 1):
            # Prepare an array to house the flattened coefficient matrices for each combination of
            # parameter values.
            coefs = torch.empty([n_param, self.n_coefs], dtype = torch.float32);

            # Compute the losses, coefficients for each combination of parameter values.
            loss_sindy  = torch.zeros(1, dtype = torch.float32);
            loss_coef   = torch.zeros(1, dtype = torch.float32);
            for i in range(n_param):
                """"
                Get the optimal SINDy coefficients for the i'th combination of parameter values. 
                Remember that Latent_States[i][0] is a tensor of shape (n_t(j), n_z) whose (j, k) 
                entry holds the k'th component of the j'th frame of the latent trajectory for the 
                i'th combination of parameter values. 
                
                Note that Result a 3 element tuple.
                """
                result : tuple[torch.Tensor] = self.calibrate(Latent_States = [Latent_States[i]], 
                                                              t_Grid        = [t_Grid[i]]);

                # Package the results from this combination of parameter values.
                coefs[i, :] = result[0];
                loss_sindy += result[1];
                loss_coef  += result[2];
            
            # Package everything to return!
            return coefs, loss_sindy, loss_coef;
        

        # -----------------------------------------------------------------------------------------
        # Evaluate for one combination of parameter values case.

        Z       : torch.Tensor  = Latent_States[0];
        t_Grid  : torch.Tensor  = t_Grid[0];
        n_t     : int           = len(t_Grid);

        Z_D     : torch.Tensor  = Z[0];
        Z_V     : torch.Tensor  = Z[1];
        
        # First, compute the second time derivative of Z_D. This should also be the first time 
        # derivative of Z_V. We average the two so that the final loss depends on both.
        if(self.Uniform_t_Grid  == True):
            h : float = t_Grid[1] - t_Grid[0];
            d2Z_dt2_from_Z_D    : torch.Tensor  = Derivative2_Order4(X = Z_D,   h = h);
            d2Z_dt2_from_Z_V    : torch.Tensor  = Derivative1_Order4(X = Z_V,   h = h);
        else:
            d2Z_dt2_from_Z_D    : torch.Tensor  = Derivative2_Order2_NonUniform(X = Z_D, t_Grid = t_Grid);
            d2Z_dt2_from_Z_V    : torch.Tensor  = Derivative1_Order2_NonUniform(X = Z_V, t_Grid = t_Grid);
        d2Z_dt2             : torch.Tensor  = 0.5*(d2Z_dt2_from_Z_D + d2Z_dt2_from_Z_V);

        # Concatenate Z_D, Z_V and a column of 1's. We will solve for the matrix, E, which gives 
        # the best fit for the system d2Z_dt2 = cat[Z_D, Z_V, 1] E. This matrix has the form 
        # E^T = [-K, -C, b]. Thus, we can extract K, C, and b from Z_1.
        Z_1   : torch.Tensor  = torch.cat([Z_D, Z_V, torch.ones((Z_D.shape[0], 1))], dim = 1);

        # For each j, solve the least squares problem 
        #   min{ || d2Z_dt2[:, j] - Z_1 E(j)|| : E(j) \in \mathbb{R}^(n_z*(2*n_z + 1)) }
        # We store the resulting solutions in a matrix, coefs, whose j'th column holds the 
        # results for the j'th column of Z_V. Thus, coefs is a 2d tensor with shape 
        # (2*n_z + 1, n_z).
        coefs   : torch.Tensor  = torch.linalg.lstsq(Z_1, d2Z_dt2).solution;

        # Compute the losses
        Loss_LD     = self.LD_LossFunction(d2Z_dt2, torch.matmul(Z_1, coefs));
        Loss_Coef   = torch.norm(coefs, self.coef_norm_order);

        if(False):
            # Extract K, C, and b from coefs.
            E   : torch.Tensor  = coefs.T;
            K   : torch.Tensor  = -E[:, 0:self.n_z];
            C   : torch.Tensor  = -E[:, self.n_z:(2*self.n_z)];
            b   : torch.Tensor  = E[:, 2*self.n_z:(2*self.n_z + 1)];
            
            # Compute the RHS of the diff eq using coefs and the matrices we found.
            RHS_coefs           = torch.matmul(Z_1, coefs);
            RHS_Manual          = torch.matmul(torch.ones((Z_D.shape[0], 1)), b.T) - torch.matmul(Z_V, C.T) - torch.matmul(Z_D, K.T);

            # Select a random row to sample.
            import random;
            row : int           = random.randint(a = 0, b = Z_D.shape[0]);

            print("Row %d of RHS using coefs:                   %s" % (row, str(RHS_coefs[row, :])));
            print("Row %d of RHS using K, C, and b:             %s" % (row, str(RHS_Manual[row, :])));
            print("Max diff between RHS with coefs and K/C/b:   %f" % torch.max(torch.abs(RHS_coefs - RHS_Manual)));

        # Prepare coefs and the losses to return. 
        # Note: we flatten the coefficient matrix.
        coefs   : torch.Tensor  = coefs.detach().flatten();
        return coefs, Loss_LD, Loss_Coef;
    


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
        
        coefs: A two dimensional numpy.ndarray or torch.Tensor objects of shape (n_param, n_coef)
        whose i'th row represents the optimal set of coefficients when we use the i'th combination 
        of parameter values. We inductively call simulate on each row of coefs. 

        IC: An n_param element list whose i'th element is an n_IC element list whose j'th element
        is a 2d numpy.ndarray or torch.Tensor object of shape (n(i), n_z). Here, n(i) is the 
        number of initial conditions (for a fixed set of coefficients) we want to simulate forward 
        using the i'th set of coefficients. Further, n_z is the latent dimension. If you want to 
        simulate a single IC, for the i'th set of coefficients, then n(i) == 1. IC[i][j][k, :] 
        should hold the k'th initial condition for the j'th derivative of the latent state when
        we use the i'th combination of parameter values. 

        t_Grid: A n_param element list whose i'th entry is a 2d numpy.ndarray or torch.Tensor 
        object. The i'th entry should either have shape (n(i), n_t(i)) or shape (n_t(i)). Use the
        former case when we want to use different times for each initial condition and the latter
        case when we want to use the same times for all initial conditions. 
        
        In the former case, the j,k array entry specifies k'th time value at which we solve for 
        the latent state when we use the j'th initial condition and the i'th set of coefficients. 
        Each row should be in ascending order. 
        
        In the latter case, the j'th entry should specify the j'th time value at which we solve for 
        each latent state when we use the i'th combination of parameter values.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------        
        
        An n_param element list whose i'th item is a list of length n_IC whose j'th entry is a 3d 
        array of shape (n_t(i), n(i), n_z). The p, q, r entry of this array should hold the r'th 
        component of the p'th frame of the j'th tine derivative of the solution to the latent 
        dynamics when we use the q'th initial condition for the i'th combination of parameter 
        values.
        """

        # Run checks.
        assert(len(coefs.shape)     == 2);
        n_param : int = coefs.shape[0];
        assert(isinstance(t_Grid, list));
        assert(isinstance(IC, list));
        assert(len(IC)              == n_param);
        assert(len(t_Grid)          == n_param);
        
        assert(isinstance(IC[0], list));
        n_IC : int = len(IC[0]);
        assert(n_IC == 2);
        for i in range(n_param):
            assert(isinstance(IC[i], list));
            assert(len(IC[i]) == n_IC);
            assert(len(t_Grid[i].shape) == 2 or len(t_Grid[i].shape) == 1);
            for j in range(n_IC):
                assert(len(IC[i][j].shape) == 2);
                assert(type(coefs)          == type(IC[i][j]));
                assert(IC[i][j].shape[1]    == self.n_z);
                if(len(t_Grid[i].shape) == 2):
                    assert(t_Grid[i].shape[0] == IC[i][j].shape[0]);


        # -----------------------------------------------------------------------------------------
        # If there are multiple combinations of parameter values, loop through them.

        # This function behaves differently if there is one set of coefficients or multiple of them.
        if(n_param > 1):
            LOGGER.debug("Simulating with %d parameter combinations" % n_param);

            # Cycle through the parameter combinations
            Z   : list[list[numpy.ndarray]] | list[list[torch.Tensor]]  = [];
            for i in range(n_param): 
                # Fetch the i'th set of coefficients, the corresponding collection of initial
                # conditions, and the set of time values.
                ith_coefs   : numpy.ndarray             | torch.Tensor              = coefs[i, :].reshape(1, -1);
                ith_IC      : list[list[numpy.ndarray]] | list[list[torch.Tensor]]  = [IC[i]];
                ith_t_Grid  : list[numpy.ndarray]                                   = [t_Grid[i]];

                # Call this function using them. This should return a 2 element holding the 
                # displacement and velocity of the solution for the i'th combination of 
                # parameter values.
                ith_Results : list[numpy.ndarray]   | list[torch.Tensor]    = self.simulate(coefs   = ith_coefs, 
                                                                                            IC      = ith_IC, 
                                                                                            t_Grid  = ith_t_Grid)[0];

                # Add these results to X.
                Z.append(ith_Results);

            # All done.
            return Z;


        # -----------------------------------------------------------------------------------------
        # Evaluate for one combination of parameter values case.

        # In this case, there is just one parameter. Extract t_Grid, which has shape 
        # (n(i), n_t(i)) or (n_t(i)).
        t_Grid  : numpy.ndarray | torch.Tensor  = t_Grid[0];
        if(isinstance(t_Grid, torch.Tensor)):
            t_Grid = t_Grid.detach().numpy();
        n_t_i   : int           = t_Grid.shape[-1];
        if(len(t_Grid.shape) == 1):
            Same_t_Grid : bool = True;
        else:
            Same_t_Grid : bool = False;
        
        # coefs has shape (1, n_coefs). Each element of IC should have shape (n(i), n_z). 
        D0  : numpy.ndarray | torch.Tensor  = IC[0][0]; 
        V0  : numpy.ndarray | torch.Tensor  = IC[0][1];
        n_i : int                           = D0.shape[0];

        # First, we need to extract -K, -C, and b from coefs. We know that coefs is the least 
        # squares solution to d2Z_dt2 = hstack[Z, dZdt, 1] E^T. Thus, we expect that.
        # E = [-K, -C, b]. 
        E   : numpy.ndarray | torch.Tensor = coefs.reshape([2*self.n_z + 1, self.n_z]).T;

        # Extract K, C, and b. Note that we need to reshape b to have shape (1, n_z) to enable
        # broadcasting.
        K   : numpy.ndarray | torch.Tensor = -E[:, 0:self.n_z];
        C   : numpy.ndarray | torch.Tensor = -E[:, self.n_z:(2*self.n_z)];
        b   : numpy.ndarray | torch.Tensor = E[:, 2*self.n_z].reshape(1, -1);

        # Set up a lambda function to approximate (d^2/dt^2)z(t) \approx -K z(t) - C (d/dt)z(t) + b.
        # In this case, we expect dz_dt and z to have shape (n(i), n_z). Thus, matmul(z, K.T) will 
        # have shape (n(i), n_z). The i'th row of this should hold the z portion of the rhs of the 
        # latent dynamics for the i'th IC. Similar results hold for dot(dz_dt, C.T). The final 
        # result should have shape (n(i), n_z). The i'th row should hold the rhs of the latent 
        # dynamics for the i'th IC.
        if(isinstance(coefs, numpy.ndarray)):
            f   = lambda t, z, dz_dt: b - numpy.matmul(dz_dt, C.T)  - numpy.matmul(z, K.T);
        if(isinstance(coefs, torch.Tensor)):
            f   = lambda t, z, dz_dt: b - torch.matmul(dz_dt, C.T)  - torch.matmul(z, K.T);

        # Solve the ODE forward in time. D and V should have shape (n_t, n(i), n_z). If we use the 
        # same t values for each IC, then we can exploit the fact that the latent dynamics are 
        # autonomous to solve using each IC simultaneously. Otherwise, we need to run the latent
        # dynamics one IC at a time. 
        if(Same_t_Grid == True):
            D, V = RK4(f = f, y0 = D0, Dy0 = V0, t_Grid = t_Grid);
        else:
            # Set up arrays to hold the results of each simulation.
            if(isinstance(coefs, numpy.ndarray)):
                D : numpy.ndarray   = numpy.empty((n_t_i, n_i, self.n_z), dtype = numpy.float32);
                V : numpy.ndarray   = numpy.empty((n_t_i, n_i, self.n_z), dtype = numpy.float32);
            elif(isinstance(coefs, torch.Tensor)):
                D : torch.Tensor    = torch.empty((n_t_i, n_i, self.n_z), dtype = torch.float32);
                V : torch.Tensor    = torch.empty((n_t_i, n_i, self.n_z), dtype = torch.float32);
            
            # Now cycle through the ICs.
            for j in range(n_i):
                D_j, V_j    = RK4(f = f, y0 = D0[j, :], Dy0 = V0[j, :], t_Grid = t_Grid[j, :]);
                D[:, j, :]  = D_j;
                V[:, j, :]  = V_j;

        # All done!
        return [[D, V]];