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
    n_z             : int           = -1;       # Dimensionality of the latent space
    n_coefs         : int           = -1;       # Number of coefficients in the latent space dynamics
    n_IC            : int           = -1;       # Number of initial conditions to define the initial latent state.
    Uniform_t_Grid  : bool          = False;    # Is there an h such that the i'th frame is at t0 + i*h? Or is the spacing between frames arbitrary?
    coefs           : torch.Tensor  = torch.Tensor([]);



    def __init__(   self, 
                    n_z             : int,
                    Uniform_t_Grid  : bool, 
                    config          : dict) -> None:
        r"""
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

        Uniform_t_Grid : bool 
            If True, then for each parameter value, the times corresponding to the frames of the 
            solution for that parameter value will be uniformly spaced. In other words, the first 
            frame corresponds to time t0, the second to t0 + h, the k'th to t0 + (k - 1)h, etc 
            (note that h may depend on the parameter value, but it needs to be constant for a 
            specific parameter value). The value of this setting determines which finite difference 
            method we use to compute time derivatives. 

        config : dict
            The "latent_dynamics" sub-dictionary of the config file. 

            
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        Nothing!
        """

        # Set class variables.
        self.n_z             = n_z;
        self.Uniform_t_Grid  = Uniform_t_Grid;
        self.config          = config;

        # There must be at least one latent dimension and there must be at least 1 time step.
        assert(self.n_z > 0);

        # All done!
        return;



    def fit_coefficients(self,
                         Latent_States   : list[list[torch.Tensor]],
                         t_Grid          : list[torch.Tensor],
                         params          : numpy.ndarray | None = None) -> torch.Tensor:
        r"""
        Fit (initialize) latent dynamics coefficients from latent state data.

        This method is intended for **coefficient initialization** (e.g., when greedy sampling
        adds a new training parameter and we need a reasonable starting value for its coefficients).
        It should return, for each parameter combination, a 1D coefficient vector of length
        `self.n_coefs`.

        Design rule:
        - `calibrate(...)` computes the LD loss (and other regularizers) **given coefficients**.
        - `fit_coefficients(...)` estimates coefficients **from data**.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_States : list[list[torch.Tensor]], len = n_param
            The i'th list element is an `n_IC`-element list whose j'th entry is a 2D tensor of
            shape (n_t(i), n_z) containing the j'th derivative of the latent state trajectory for
            the i'th parameter combination.

        t_Grid : list[torch.Tensor], len = n_param
            The i'th element is a 1D tensor of shape (n_t(i)) holding the time grid for the i'th
            parameter combination.

        params : numpy.ndarray, shape = (n_param, n_p), optional
            The i'th row holds the i'th combination of parameter values. Some latent dynamics
            models may require these values (e.g., weak-form test-function lookup or parametric
            forcing). Default is None for models that do not use parameters.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        coefs : torch.Tensor, shape = (n_param, n_coefs)
            The i'th row holds the fitted coefficient vector for the i'th parameter combination.
        """

        raise RuntimeError("Abstract function LatentDynamics.fit_coefficients!");
    


    @staticmethod
    def stability_penalty(A: torch.Tensor, margin : float = 0.1) -> torch.Tensor:
        """
        Differentiable stability regularizer for linear systems z' = Az (+ b).

        We penalize positive growth rates by computing the largest eigenvalue of the symmetric
        part of A:  sym(A) = (A + A^T)/2.  If lambda_max(sym(A)) <= 0 then the system is
        contractive in the Euclidean norm.

        Returns a smooth nonnegative penalty: softplus(lambda_max(sym(A)) + margin).
        """

        # Checks
        assert isinstance(A, torch.Tensor), f"A must be a torch.Tensor, got {type(A)}";
        assert A.ndim == 2 and A.shape[0] == A.shape[1], f"A must be square, got {tuple(A.shape)}";

        # Compute symmetric part of A
        sym         = 0.5 * (A + A.T);

        # Now compute the maximum eigenvalue.
        lam_max     = torch.linalg.eigvalsh(sym).max();
        return torch.nn.functional.softplus(lam_max + margin);



    def calibrate(  self, 
                    Latent_States   : list[list[torch.Tensor]], 
                    loss_type       : str,
                    t_Grid          : list[torch.Tensor], 
                    params          : numpy.ndarray | None  = None,
                    input_coefs     : list[torch.Tensor]    = []) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
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
        
        loss_type : str
            The type of loss function to use. Must be either "MSE" or "MAE".

        t_Grid : list[troch.Tensor], len = n_param
            The i'th element should be a 1d tensor of shape (n_t(i)) whose j'th element holds the 
            time value corresponding to the j'th frame when we use the i'th combination of 
            parameter values.

        input_coefs : list[torch.Tensor], len = n_param
            The i'th element of this list is a 1d tensor of shape (n_coefs) holding the
            coefficients for the i'th combination of parameter values. This function assumes
            coefficients are provided; to *fit* coefficients from data, use `fit_coefficients(...)`.

        params : numpy.ndarray, shape = (n_param, n_p), optional
            The i'th row holds the i'th combination of parameter values. This can be used by latent 
            dynamics models that depend explicitly on parameter values (e.g., for time-varying or 
            parameterized forcing). Default is None for latent dynamics that don't use parameters.

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        coefs, loss_sindy, loss_stab. 
        
        coefs : torch.Tensor, shape = (n_param, n_coef)
            A matrix of shape (n_param, n_coef). The i,j entry of this array holds the value of 
            the j'th coefficient when we use the i'th combination of parameter values.

        loss_sindy : list[torch.Tensor], len = n_param
            The i'th element of this list is a 0-dimensional tensor whose lone element holds the 
            sum of the SINDy losses from the i'th combination of parameter values. 

        loss_coef : list[torch.Tensor], len = n_para
            The i'th element of this list is a 0-dimensional tensor whose lone element holds the
            coefficient loss (Frobenius norm) of the coefficients for the i'th combination 
            of parameter values.      
            
        loss_stab : list[torch.Tensor], len = n_param
            The i'th element of this list is a 0-dimensional tensor whose lone element holds the
            coefficient regularization term for the i'th combination of parameter values. In the
            current codebase this is a *stability penalty* on the learned linear dynamics matrix
            (see LatentDynamics.stability_penalty).
        """

        raise RuntimeError('Abstract function LatentDynamics.calibrate!');
    


    def simulate(   self,
                    coefs   : numpy.ndarray             | torch.Tensor, 
                    IC      : list[list[numpy.ndarray   | torch.Tensor]],
                    t_Grid  : list[numpy.ndarray        | torch.Tensor],
                    params  : numpy.ndarray | None = None) -> list[list[numpy.ndarray | torch.Tensor]]:
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
        
        params : numpy.ndarray, shape = (n_param, n_p), optional
            The i'th row holds the i'th combination of parameter values. This can be used by latent 
            dynamics models that depend explicitly on parameter values (e.g., for time-varying or 
            parameterized forcing). Default is None for latent dynamics that don't use parameters.

     
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
                      'config'          : self.config,
                      'Uniform_t_Grid'  : self.Uniform_t_Grid};
        return param_dict;



    def load(self, dict_ : dict) -> None:
        assert(self.n_z             == dict_['n_z']);
        assert(self.n_coefs         == dict_['n_coefs']);
        assert(self.n_IC            == dict_['n_IC']);
        assert(self.Uniform_t_Grid  == dict_['Uniform_t_Grid']);
        return;
    
