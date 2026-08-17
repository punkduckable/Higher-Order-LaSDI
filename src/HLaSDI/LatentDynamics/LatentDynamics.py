# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  logging;

import  numpy;
import  torch;
from    pydantic            import  BaseModel, ConfigDict, model_validator;

from    HLaSDI.Schemas     import   LatentDynamicsBaseConfig


# Logger setup.
LOGGER : logging.Logger = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# Schemas
# -------------------------------------------------------------------------------------------------

class LD_Loss_Container(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed = True);

    # A dictionary (with the same keys as weights) housing the losses. Each value is either a 
    # list (of length n_param, this is used for parameter-dependant losses) of single element 
    # tensors or a single element tensor holding the losses.
    losses: dict[str, torch.Tensor | list[torch.Tensor]];

    # A dictionary (same keys as losses) holding the loss weights
    weights: dict[str, float];

    # The parameters used to compute the losses stored in this object. [n_param, n_p]
    params : numpy.ndarray;      

    @model_validator(mode = "after")
    def validate_activations_and_active_count(self) -> "LD_Loss_Container":
        # Ensure both dicts have the same keys.
        if set(self.losses.keys()) != set(self.weights.keys()):
            raise ValueError("Losses and weights must have the same keys! losses.keys() = %s, weights.keys() = %s" % (str(self.losses.keys()), str(self.weights.keys()) ));

        # Ensure params is a 2d numpy array
        assert len(self.params.shape) == 2, "params must be a 2d numpy.ndarray, got shape %s" % str(self.params.shape);
        n_param : int = self.params.shape[0];

        # Make sure each loss is a tensor or n_param list of tensors.
        for key, value in self.losses.items():
            assert isinstance(key, str), "all losses keys must be strings, but one key (%s) has type %s!" % (str(key), str(type(key)));
            assert isinstance(value, torch.Tensor) or isinstance(value, list), "each loss must be a tensor or list of tensors, losses[%s] has type %s" % (key, str(type(value)));
            if isinstance(value, torch.Tensor):
                assert value.numel() == 1, "each loss must have a single element, losses[%s] has shape %s" % (key, str(value.shape));
            else: 
                assert len(value) == n_param, "Each list item of a LD_Loss_Container object must have length %d, but losses[%s] has length %d" % (n_param, key, len(value));
                for idx, item in enumerate(value):
                    assert isinstance(item, torch.Tensor), "Each loss must be a tensor, but losses[%s][%d] has type %s" % (key, idx, type(item));
                    assert item.numel() == 1, "each loss must have a single element, losses[%s][%d] has shape %s" % (key, idx, str(item.shape));

        # Make sure all loss weights are floats
        for key, value in self.weights.items():
            assert isinstance(key, str), "all weights keys must be strings, but one key (%s) has type %s!" % (str(key), str(type(key)));
            assert isinstance(value, float), "all weights values must be floats, but losses[%s] has type %s!" % (key, str(type(value)));

        # All done :) 
        return self;

# -------------------------------------------------------------------------------------------------
# LatentDynamics base class
# -------------------------------------------------------------------------------------------------

class LatentDynamics:
    r"""
    This is the base interface for parameterized latent dynamics.

    In the HLaSDI framework, a ROM consists of an EncoderDecoder model and a LatentDynamics 
    object (acting as the Encoder/Decoder and Latent Dynamics portions of the ROM, respectively). 
    These are jointly trained via a Trainer object using data from a Physics object. 
    
    A Sampler object determines how the model picks which testing example to ad to the 
    training set after each round of training.

    A `LatentDynamics` subclass defines an ODE model for the time evolution of the latent 
    encodings in an EncoderDecoder model. i.e., this defines the actual LatentDynamics in the 
    LaSDI model. 

    LatentDynamics models can rollout latent trajectories (via the simulate method) by solving 
    the latent ODE associated with a particular parameter value, compute the Latent Dynamics, 
    coefficient, and stability losses associated with a collection of parameter values (via the 
    compute_losses method), or fit a set of coefficients to a time series of latent states for a 
    particular parameter value (via the "initialize_coefficients" method).

    Interpolatable latent-dynamics objects store the latent dynamics coefficients (learnable
    parameters that define the latent dynamics model) for the training set. These are stored in
    the `train_coefs` attribute added by `InterpolatableLatentDynamics`, which is a dictionary that
    uses a parameter value as the key and a "coefficient dictionary" as its associated value. Each
    coefficient dictionary should itself be a dictionary with string keys and tensor values; each
    item is associated with one of the matrices or vectors in the latent dynamic model (e.g.,
    {"A" : A, "b" : b"} would be a typical coefficient dictionary for a SINDy model, where A is
    the system matrix and b is the bias vector in the SINDy latent dynamics model z' = Az + b).

    

    -----------------------------------------------------------------------------------------------
    Class/instance variables
    -----------------------------------------------------------------------------------------------

    n_z : int
        Latent-space dimension.  Each latent state component has length `n_z`.
    
    n_IC : int
        Number of latent initial-condition components required to start the dynamics.  For example,
        first-order dynamics typically use `n_IC = 1`, while second-order dynamics use position and
        velocity components with `n_IC = 2`.
    
    n_p : int 
        The number of (scalar) parameters in the parameter space.

    Uniform_t_Grid : bool
        Whether each trajectory's time grid is uniformly spaced; subclasses use this to choose
        appropriate finite-difference or weak-form derivative approximations.
    
    stochastic : bool
        Indicates if the latent dynamics outside of the train set stochastic or deterministic.
        Determines which sampling routines we can use, among other things.

    trainable : bool 
        Indicates if the trainer should train the parameters in this LatentDynamics object. 
        Sub-classes should configure `parameters` to return an empty list if 
        `trainable = False`. 

    loss_weights : dict[str, NonNegativeFloat]
        A dictionary housing the weights of the losses computed by `compute_losses`. Its keys are
        the latent-dynamics loss API for this subclass: `compute_losses` must return exactly these
        names, and trainers use the values here when forming the total objective.
        
    config : LatentDynamicsBaseConfig
        The validated `latent_dynamics` configuration object used to construct the concrete model.


    -----------------------------------------------------------------------------------------------
    Subclassing
    -----------------------------------------------------------------------------------------------
    To define a new latent-dynamics model, subclass `LatentDynamics`, call the base-class 
    initializer, and define the following:

    - `initialize_coefficients(Latent_States, t_Grid, params=None)`: estimate/initialize native
      coefficient dictionaries from encoded trajectories.

    - `parameters()`: return the actual parameters (if training is enabled) stored in 
    `train_coefs` so the `Trainer` can optimize them jointly with the encoder/decoder.
    
    - `compute_losses(Latent_States, t_Grid, step, params=None)`: compute latent-dynamics losses for
      the current coefficients and return an `LD_Loss_Container` whose loss keys match
      `self.loss_weights`. Values may be scalar tensors for global losses or length-`n_param` lists
      of scalar tensors for per-parameter losses. The loss metric (e.g., MSE vs MAE) is a subclass
      implementation detail; there is no trainer-level `loss_type` argument for latent dynamics.
    
    - `simulate(IC, t_Grid, params, sample=False)`: integrate the latent ODE from one latent
      initial condition per parameter value and return latent trajectories in the expected
      `n_IC`-component format.

    - `RHS(Z, t_Grid, params, sample=False)`: evaluate the pointwise right-hand side of the latent
      ODE at supplied latent states. Strong and weak forms of the same latent ODE should normally
      use the same RHS implementation because the weak/strong distinction changes only how
      residual losses are computed.

    - `export`, `load`: export
    """
    # Instance variables
    n_z             : int;          # Dimensionality of the latent space
    n_IC            : int;          # Number of initial conditions to define the initial latent state.
    n_p             : int;          # The number of parameters in the parameter space.
    Uniform_t_Grid  : bool;         # Is there an h such that the i'th frame is at t0 + i*h? Or is the spacing between frames arbitrary?
    trainable       : bool          # Should the trainer train the latent dynamics parameters?
    stochastic      : bool          # Are the latent dynamics outside of the train set stochastic or deterministic?
    loss_weights    : dict          # Dictionary housing loss weights; also specifies loss names.
    config          : LatentDynamicsBaseConfig  # The validated latent_dynamics configuration object used to define the LatentDynamics object


    def __init__(   self, 
                    n_z             : int,
                    n_IC            : int, 
                    n_p             : int,
                    Uniform_t_Grid  : bool, 
                    trainable       : bool,
                    stochastic      : bool,
                    config          : LatentDynamicsBaseConfig) -> None:
        r"""
        Initializes a LatentDynamics object. Each LatentDynamics object needs to have a 
        dimensionality (n_z), a number of time steps, a model for the latent space dynamics, and 
        set of coefficients for that model. The model should describe a set of ODEs in 
        \mathbb{R}^{n_z}. These ODEs should contain a set of unknown coefficients. We learn those 
        coefficients using the compute_losses function. Once we have learned the coefficients, we 
        can solve the corresponding set of ODEs forward in time using the simulate function.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        n_z : int
            The number of dimensions in the latent space, where the latent dynamics takes place.

        n_IC : int
            Number of latent initial-condition components required to start the dynamics. For 
            example, first-order dynamics typically use `n_IC = 1`, while second-order dynamics use 
            position and velocity components with `n_IC = 2`.
    
        n_p : int 
            The number of (scalar) parameters in the parameter space.

        Uniform_t_Grid : bool 
            If True, then for each parameter value, the times corresponding to the frames of the 
            solution for that parameter value will be uniformly spaced. In other words, the first 
            frame corresponds to time t0, the second to t0 + h, the k'th to t0 + (k - 1)h, etc 
            (note that h may depend on the parameter value, but it needs to be constant for a 
            specific parameter value). The value of this setting determines which finite difference 
            method we use to compute time derivatives. 
        
        trainable : bool
            Indicates if the trainer should train the latent dynamics parameters. If false, 
            `parameters` should return an empty list.

        config : LatentDynamicsBaseConfig
            The "latent_dynamics" sub-dictionary of the config file. If `type == "weak"`, the
            model-specific sub-dictionary `config[config["type"]]` must contain `overlap`,
            `test_func_width`, and `test_func_type`.

            
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        Nothing!
        """

        assert isinstance(config, LatentDynamicsBaseConfig), "Config must be a LatentDynamicsBaseConfig, not %s" % str(type(config));

        # Set class variables.
        self.n_z             = n_z;
        self.n_IC            = n_IC;
        self.n_p             = n_p;
        self.Uniform_t_Grid  = Uniform_t_Grid;
        self.trainable       = trainable;
        self.stochastic      = stochastic;
        self.loss_weights    = config.loss_weights;
        self.config          = config;

        # There must be at least one latent dimension and there must be at least 1 time step.
        assert(self.n_z > 0);
        assert(self.n_IC > 0);

        # All done!
        return;



    # ---------------------------------------------------------------------------------------------
    # Fit Coefficients (compute initial coefficients for a particular training parameter).
    # ---------------------------------------------------------------------------------------------

    def initialize_coefficients(
            self,
            Latent_States   : list[list[torch.Tensor]],
            t_Grid          : list[torch.Tensor],
            device          : torch.device,
            params          : numpy.ndarray) -> None:
        r"""
        Fit (initialize) latent dynamics coefficients from latent state data.

        This method is intended for **coefficient initialization** (e.g., when greedy sampling
        adds a new training parameter and we need a reasonable starting value for its coefficients).

        Design rule:
        - `compute_losses(...)` computes the LD loss (and other regularizers) **given 
        coefficients**.
        - `initialize_coefficients(...)` estimates coefficients **from data**.


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
        
        device : torch.device
            The device where we want to store the new coefficients.
                
        params : numpy.ndarray, shape = (n_param, n_p), optional
            The i'th row holds the i'th combination of parameter values. Some latent dynamics
            models may require these values (e.g., weak-form test-function lookup or parametric
            forcing).


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        None. 
        """

        raise RuntimeError("Abstract function LatentDynamics.initialize_coefficients!");
    

    
    @staticmethod
    def _param_key(params_row : numpy.ndarray | torch.Tensor | list | tuple) -> tuple[float, ...]:
        r"""
        Convert one row of parameter values into a tuple.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        params_row : numpy.ndarray or torch.Tensor or list or tuple
            A one-dimensional collection of parameter values. If a 2D row-like array/tensor is
            supplied, it is flattened before conversion.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        key : tuple[float, ...]
            A hashable tuple of Python floats. 
        """
        if isinstance(params_row, torch.Tensor):
            params_row = params_row.detach().cpu().reshape(-1).tolist();
        elif isinstance(params_row, numpy.ndarray):
            params_row = params_row.reshape(-1).tolist();
        else:
            params_row = list(params_row);
        return tuple(float(x) for x in params_row);


    def parameters(self) -> list[torch.Tensor]:
        r"""
        Return the trainable coefficient tensors owned by this LatentDynamics object.

        Concrete subclasses define the ordering because the native coefficient names and tensor
        counts differ between latent-dynamics models. The returned tensors should be the actual
        tensors stored in `self.train_coefs`, not detached copies, so they can be passed directly to
        torch optimizers.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        None.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        tensors : list[torch.Tensor]
            A list containing all trainable coefficient tensors stored in `self.train_coefs`.
        """

        raise RuntimeError("Abstract function LatentDynamics.parameters!");


    def move_parameters_to_device(self, device : torch.device | str) -> None:
        r"""
        Move LD-owned parameters to a device.

        This hook exists so trainers do not need to know how a concrete LatentDynamics subclass
        stores its parameters. Subclasses that own parameters should override this method and 
        update their internal tensor references in-place before optimizer construction.

        The base implementation is a no-op for latent-dynamics classes with no LD-owned trainable
        tensor state.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        device : torch.device or str
            The destination device for LD-owned trainable tensor state.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        return;




    # ---------------------------------------------------------------------------------------------
    # Stability penalty method for computing stability losses
    # ---------------------------------------------------------------------------------------------

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




    # ---------------------------------------------------------------------------------------------
    # compute_losses: Compute losses for a particular set of training parameter values.
    # ---------------------------------------------------------------------------------------------
    
    def compute_losses(  
        self, 
        Latent_States   : list[list[torch.Tensor]], 
        t_Grid          : list[torch.Tensor], 
        step            : int,
        params          : numpy.ndarray | None  = None
    ) -> LD_Loss_Container:
        """
        The user must implement this class on any latent dynamics sub-class. Each latent dynamics 
        object should implement a parameterized model for the dynamics in the latent space. A 
        Latent_Dynamics object should pair each combination of parameter values with a set of 
        coefficients in the latent space. Using those parameters, we compute the losses for this 
        latent dynamics model. The losses are packaged into a dictionary and returned. 

        Note that the keys in this dictionary will match those in `self.loss_weights`. In general, 
        each value can be a single element torch.Tensor or a list (of length n_param) of single 
        element tensors. The former is reserved for when a loss is parameter-independent, and the 
        latter is for when a loss is a function of the parameter; trainers should sum the latter
        before incorporating them.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_States : list[list[torch.Tensor]], len = n_param
            The i'th list element should be an n_IC element list whose j'th element is a 2d numpy 
            array of shape (n_t(i), n_z) whose p, q element holds the q'th component of the j'th 
            derivative of the latent state during the p'th time step (whose time value corresponds 
            to the p'th element of t_Grid) when we use the i'th combination of parameter values. 
        
        t_Grid : list[torch.Tensor], len = n_param
            The i'th element should be a 1d tensor of shape (n_t(i)) whose j'th element holds the 
            time value corresponding to the j'th frame when we use the i'th combination of 
            parameter values.

        step : int
            Optimizer step number. 

        params : numpy.ndarray, shape = (n_param, n_p), optional
            The i'th row holds the i'th combination of parameter values. This can be used by latent 
            dynamics models that depend explicitly on parameter values (e.g., for time-varying or 
            parameterized forcing). Default is None for latent dynamics that don't use parameters.

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        losses : LD_Loss_Container
            A LD_Loss_Container object housing the losses and their weights. 
        """

        raise RuntimeError('Abstract function LatentDynamics.compute_losses!');
    

    # ---------------------------------------------------------------------------------------------
    # RHS: Evaluate the right hand side of the latent dynamics for a particular parameter.
    # ---------------------------------------------------------------------------------------------

    def RHS(    self, 
                Z       : list[list[torch.Tensor | numpy.ndarray]], 
                t_Grid  : list[numpy.ndarray | torch.Tensor],
                params  : numpy.ndarray,
                sample  : bool = False) -> list[torch.Tensor | numpy.ndarray]:
        """
        Evaluate the RHS of the latent dynamics at a set of latent states, times, and parameters. 

        Specifically, we assume that Z, t_Grid, and params have n_param elements. For each 
        parameter value, theta, we evaluate the right hand side of the latent dynamics for theta 
        at each time in t_Grid[i]. That is, we compute

            f(Z[i][0][k, :], Z[i][1][k, :], ... Z[i][n_IC][k, :], t_Grid[i][k], params[k, :])
        
        Where f denotes the right hand side of the latent dynamics;

            D^{(n_IC)} z(t) = f(z(t), z'(t), ... , D^{(n_IC - 1)} z(t), t, \theta)
        
        We compute this quantity for each time and parameter value, returning the results in a 
        list of lists.
        

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Z : list[list[torch.Tensor]], len = n_param
            i'th element is an list of length n_IC whose j'th element is a tensor of shape 
            [n_t(i), n_z] or [n_t(i), n_batch(i), n_z], where n_t(i) = len(t_Grid[i]). The k'th
            time slice should represent the j'th time derivative of the latent state
            corresponding to i'th parameter combination at the k'th time step.

        t_Grid : list[numpy.ndarray | torch.Tensor], len = n_param
            i'th element is a numpy.ndarray or torch.Tensor of shape [n_t(i)] whose j'th element
            holds the time when the latent state for parameter i was (Z[i][0][k, :], ... 
            Z[i][n_IC][k, :]).
        
        params : numpy.ndarray, shape = (n_param, n_p)
            Parameters at corresponding to the latent solutions stored in Z.
        
        sample : bool
            If True, we draw a sample of the latent dynamics at each parameter value to compute 
            the right hand sides. Otherwise, we use the mean.
       
        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        RH_Sides : list[numpy.ndarray | torch.Tensor], len = n_param
            i'th element is a numpy.ndarray or torch.Tensor with the same leading dimensions and
            backend as Z[i][0] and last dimension n_z. Its j'th time slice holds the right-hand
            side of the sampled (or mean) latent dynamics at params[i, :] evaluated at
            Z[i][0][j, ...], Z[i][1][j, ...], ... Z[i][n_IC - 1][j, ...], t_Grid[i][j],
            params[i, :]. For first-order dynamics this is z'; for second-order dynamics this is
            z''.
        """

        raise RuntimeError('Abstract function LatentDynamics.RHS!');


    # ---------------------------------------------------------------------------------------------
    # Simulate: Solve the latent dynamics for a specific set of training parameters.
    # ---------------------------------------------------------------------------------------------

    def simulate(   self,
                    IC      : list[list[numpy.ndarray   | torch.Tensor]],
                    t_Grid  : list[numpy.ndarray        | torch.Tensor],
                    params  : numpy.ndarray, 
                    sample  : bool = False) -> list[list[numpy.ndarray | torch.Tensor]]:
        """
        Time integrates the latent dynamics from one initial condition for each parameter value.
        Note that if self is not stochastic, we should generally not allow sampling.
 

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        IC : list[list[numpy.ndarray]] or list[list[torch.Tensor]], len = n_param
            i'th element is an n_IC element list whose j'th element is a 1d numpy.ndarray or 
            torch.Tensor object of shape (n_z), where n_z is the latent dimension. IC[i][j] should
            hold the initial conditions for the j'th derivative of the latent state when we use the 
            i'th combination of parameter values. 

        t_Grid : list[numpy.ndarray] or list[torch.Tensor], len = n_param
            i'th entry is a 1d tensor of shape (n_t(i)). The j'th entry should specify the j'th 
            time value at which we solve for each latent state when we use the i'th combination of 
            parameter values.
        
        params : numpy.ndarray, shape = (n_param, n_p)
            The i'th row holds the i'th combination of parameter values. This can be used by latent 
            dynamics models that depend explicitly on parameter values (e.g., for time-varying or 
            parameterized forcing).

        sample : bool 
            If self is stochastic, setting this to true will sample from the posterior distribution 
            of the latent dynamics at each parameter value, then solve the latent dynamics using 
            the resulting sample. Otherwise, setting this to true will use the mean of that 
            posterior distribution. If self is not stochastic, this does nothing.

     
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------        
        
        Z : list[list[numpy.ndarray]] or list[list[torch.Tensor]], len = n_parm
            i'th element is a list of length n_IC whose j'th entry is a 2d array/tensor of shape
            (n_t(i), n_z). The p, q entry of this array should hold the q'th component of the
            p'th frame of the j'th time derivative of the solution for the i'th parameter value.
        """

        raise RuntimeError('Abstract function LatentDynamics.simulate!');
