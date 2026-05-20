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
    r"""
    This is the base interface for parameterized latent dynamics.

    In the HLaSDI framework, a ROM consists of an EncoderDecoder model and a LatentDynamics 
    object (acting as the Encoder/Decoder and Latent Dynamics portions of the ROM, respectively). 
    These are jointly trained via a Trainer object using data from a Physics object. The 
    LatentDynamics object holds the learnedLatentDynamics coefficients for the training set,
    while an Interpolate object samples LatentDynamics coefficients for testing parameter 
    combinations. A Sampler object determines how the model picks which testing example to add
    to the training set after each round of training.

    A `LatentDynamics` subclass defines an ODE model for the time evolution of the latent 
    encodings in an EncoderDecoder model. i.e., this defines the actual LatentDynamics in the 
    LaSDI model. 

    LatentDynamics models can rollout latent trajectories (via the simulate method) by solving 
    the latent ODE associated with a particular parameter value, compute the Latent Dynamics, 
    coefficient, and stability losses associated with a collection of parameter values (via the 
    calibrate method), or fit a set of coefficients to a time series of latent states for a 
    particular parameter value (via the "fit_coefficients" method).

    LatentDynamics objects also store the latent dynamics coefficients (learnable parameters 
    that define the latent dynamics model) for the training set. These are stored in the 
    train_coefs attribute, which is a dictionary that uses a parameter value as the key and a 
    "coefficient dictionary" as its associated value. Each coefficient dictionary should itself 
    be a dictionary with string keys and tensor values; each item is associated with one of the 
    matrices or vectors in the latent dynamic model (e.g., {"A" : A, "b" : b"} would be a typical
    coefficient dictionary for a SINDy model, where A is the system matrix and b is the bias vector
    in the SINDy latent dynamics model z' = Az + b).

    

    -----------------------------------------------------------------------------------------------
    Class/instance variables
    -----------------------------------------------------------------------------------------------
    n_z : int
        Latent-space dimension.  Each latent state component has length `n_z`.
    
    n_coefs : int
        Number of scalar coefficients in the concrete latent-dynamics model, mainly used for
        compatibility with flattened coefficient outputs.
    
    n_IC : int
        Number of latent initial-condition components required to start the dynamics.  For example,
        first-order dynamics typically use `n_IC = 1`, while second-order dynamics use position and
        velocity components with `n_IC = 2`.
    
    Uniform_t_Grid : bool
        Whether each trajectory's time grid is uniformly spaced; subclasses use this to choose
        appropriate finite-difference or weak-form derivative approximations.
    
    config : dict
        The `latent_dynamics` configuration dictionary used to construct the concrete model.

    type : str
        Latent-dynamics formulation type. Strong formulations use pointwise ODE residuals, while
        weak formulations use compactly supported weight functions.

    weight_function_derivatives : list[dict[tuple[float, ...], torch.Tensor]]
        Weak-form weight-function derivative tensors indexed first by derivative order and then by
        parameter tuple. Entry `k` stores the `k`'th time derivatives of the weight functions.

    train_coefs : dict[tuple[float, ...], dict[str, torch.Tensor]]
        Trainable, native coefficient dictionaries indexed by parameter tuple. The training 
        parameter value (as returned by the _param_key method) is the key, while the value is a 
        dictionary housing the associated coefficients. The dictionary for a particular parameter 
        value should use string keys (corresponding to the symbols used for various matrices and
        vectors in the latent dynamics model) and tensor value. For instance, for each combination
        of parameter values in the SINDy class, the associated coefficient dictionary has two 
        keys, "A" and "b", whose values correspond to the system matrix and bias vector in the
        SINDy latent dynamics model (z' = Az + b). This should only be used to store the TRAINING
        coefficients; test values should be determined by an Interpolate object.

        
    
    -----------------------------------------------------------------------------------------------
    Subclassing
    -----------------------------------------------------------------------------------------------
    To define a new latent-dynamics model, subclass `LatentDynamics`, call `super().__init__(...)`,
    set `self.n_IC` and `self.n_coefs`, and implement:

    - `fit_coefficients(Latent_States, t_Grid, params=None)`: estimate/initialize native
      coefficient dictionaries from encoded trajectories and store them with `set_train_coefs(...)`.

    - `trainable_coef_tensors()`: return the actual trainable tensors stored in `train_coefs` so
      the `Trainer` can optimize them jointly with the encoder/decoder.
    
    - `calibrate(Latent_States, loss_type, t_Grid, params=None)`: compute latent-dynamics residual
      losses and coefficient/stability regularization for the current coefficients.
    
    - `simulate(coefs, IC, t_Grid, params=None)`: integrate the latent ODE from one or more latent
      initial conditions and return latent trajectories in the expected `n_IC`-component format.

    Subclasses may also override `flatten_coefficients(...)` if their native coefficient
    dictionaries need a specific legacy ordering for plotting or diagnostics.
    """
    # Instance variables
    n_z             : int;          # Dimensionality of the latent space
    n_coefs         : int;          # Number of coefficients in the latent space dynamics
    n_IC            : int;          # Number of initial conditions to define the initial latent state.
    Uniform_t_Grid  : bool;         # Is there an h such that the i'th frame is at t0 + i*h? Or is the spacing between frames arbitrary?
    config          : dict          # The "latent_dynamics" sub-dictionary of the configuration file, used to define the LatentDynamics object
    type            : str;          # Latent-dynamics formulation type: "strong" or "weak".
    train_coefs     : dict[tuple[float, ...], dict[str, torch.Tensor]];
    weight_function_derivatives : list[dict[tuple[float, ...], torch.Tensor]];


    def __init__(   self, 
                    n_z             : int,
                    n_coefs         : int,
                    n_IC            : int, 
                    Uniform_t_Grid  : bool, 
                    config          : dict,
                    type            : str = "strong") -> None:
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

        n_coefs : int
            An integer housing the number of coefficients in the latent dynamics model; typically 
            (# of matrices in the LD model)*n_z^2 + (# of vectors in the LD model)*n_z

        n_IC : int
            Number of latent initial-condition components required to start the dynamics. For 
            example, first-order dynamics typically use `n_IC = 1`, while second-order dynamics use 
            position and velocity components with `n_IC = 2`.
    
        Uniform_t_Grid : bool 
            If True, then for each parameter value, the times corresponding to the frames of the 
            solution for that parameter value will be uniformly spaced. In other words, the first 
            frame corresponds to time t0, the second to t0 + h, the k'th to t0 + (k - 1)h, etc 
            (note that h may depend on the parameter value, but it needs to be constant for a 
            specific parameter value). The value of this setting determines which finite difference 
            method we use to compute time derivatives. 

        config : dict
            The "latent_dynamics" sub-dictionary of the config file. If `type == "weak"`, the
            model-specific sub-dictionary `config[config["type"]]` must contain `overlap`,
            `test_func_width`, and `test_func_type`.

        type : str, optional
            The latent-dynamics formulation type. Must be either "strong" or "weak". Strong
            formulations compare pointwise ODE residuals, while weak formulations use compactly
            supported weight functions and their time derivatives. Default is "strong".

            
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        Nothing!
        """

        # Set class variables.
        self.n_z             = n_z;
        self.n_coefs         = n_coefs;
        self.n_IC            = n_IC;
        self.Uniform_t_Grid  = Uniform_t_Grid;
        self.config          = config;
        self.type            = type;
        self.train_coefs     : dict[tuple[float, ...], dict[str, torch.Tensor]] = {};
        self.weight_function_derivatives : list[dict[tuple[float, ...], torch.Tensor]] = [
            {} for _ in range(self.n_IC + 1)
        ];

        # There must be at least one latent dimension and there must be at least 1 time step.
        assert(self.n_z > 0);
        assert(self.n_IC > 0);
        assert self.type in ["strong", "weak"], "LatentDynamics.type must be either 'strong' or 'weak'";

        # Weak-form settings are owned by the base class so all weak LatentDynamics subclasses use
        # the same test-function construction and lookup path.
        self.test_func_type  : str   | None = None;
        self.test_func_width : float | None = None;
        self.overlap         : float | None = None;
        self.pq              : int   | None = None;
        if self.type == "weak":
            # Weak form specific checks
            assert isinstance(config, dict),    "Weak LatentDynamics requires a config dictionary";
            assert "type" in config,            "Weak LatentDynamics config must contain the model selector key 'type'";
            model_type  : str   = config["type"];
            assert model_type in config,        "Weak LatentDynamics config must contain config[config['type']]";
            weak_config : dict  = config[model_type];
            for key in ["overlap", "test_func_width", "test_func_type"]:
                assert key in weak_config,          "Weak LatentDynamics config[%s] must contain '%s'" % (model_type, key);
            
            # Weak form setup. 
            self.test_func_type  = weak_config["test_func_type"];
            self.test_func_width = float(weak_config["test_func_width"]);
            self.overlap         = float(weak_config["overlap"]);
            self.pq              = self.n_IC + 1;

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

        None. Subclasses store native coefficient dictionaries in `self.train_coefs`.
        """

        raise RuntimeError("Abstract function LatentDynamics.fit_coefficients!");
    


    @staticmethod
    def _param_key(params_row : numpy.ndarray | torch.Tensor | list | tuple) -> tuple[float, ...]:
        r"""
        Convert one row of parameter values into the exact key used by `train_coefs`.

        This helper intentionally does not round values or perform fuzzy matching. The parameter
        tuple returned here must match the key stored in `self.train_coefs`; otherwise, subsequent
        dictionary lookups should raise a KeyError.


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
            A hashable tuple of Python floats. This tuple is used as the key in `self.train_coefs`.
        """
        if isinstance(params_row, torch.Tensor):
            params_row = params_row.detach().cpu().reshape(-1).tolist();
        elif isinstance(params_row, numpy.ndarray):
            params_row = params_row.reshape(-1).tolist();
        else:
            params_row = list(params_row);
        return tuple(float(x) for x in params_row);



    def _get_support_intervals( self,
                                T : float,
                                L : float,
                                s : float) -> tuple[numpy.ndarray, numpy.ndarray]:
        r"""
        Generate support intervals for compactly supported weak-form weight functions.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        T : float
            Final time value. The generated intervals lie in `[0, T]`.

        L : float
            Support width for each weight function.

        s : float
            Overlap amount between adjacent supports. The distance between adjacent left endpoints
            is `L - s`.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        a_s, b_s : tuple[numpy.ndarray, numpy.ndarray]
            One-dimensional arrays holding the left and right endpoints of each support interval.
        """

        assert float(T) > 0.0, "T must be positive";
        assert float(L) > 0.0, "L must be positive";
        assert float(s) >= 0.0, "s must be nonnegative";
        assert float(s) < float(L), "overlap amount s must be smaller than support width L";
        assert float(L) <= float(T), "test-function support width L must be no larger than T";

        grid : list[list[float]] = [];
        a : float = 0.0;
        b : float = float(L);
        grid.append([a, b]);
        while (b - float(s) + float(L)) <= float(T):
            a = b - float(s);
            b = a + float(L);
            grid.append([a, b]);

        grid_array = numpy.asarray(grid, dtype = numpy.float64);
        return grid_array[:, 0], grid_array[:, 1];



    def _weak_weight_function(self,
                              t : torch.Tensor,
                              a : float,
                              b : float) -> torch.Tensor:
        r"""Evaluate one weak-form weight function on `t`."""

        assert self.test_func_type is not None;
        if self.test_func_type == "bump":
            eta     : float = 5.0;
            half_L  : float = 0.5 * (float(b) - float(a));
            center  : float = 0.5 * (float(a) + float(b));
            const   : float = eta;
            nugget  : float = 1.0e-7;
            a_space = numpy.linspace(-half_L + nugget, half_L - nugget, 1000);
            bump    = numpy.exp(-eta / (1.0 - (a_space / half_L) ** 2));
            C       : float = float(1.0 / numpy.trapz(bump, a_space) / numpy.exp(const));

            x           : torch.Tensor = (t - center) / half_L;
            denom       : torch.Tensor = 1.0 - x ** 2;
            inside      : torch.Tensor = denom > 0.0;
            safe_denom  : torch.Tensor = torch.where(inside, denom, torch.ones_like(denom));
            values      : torch.Tensor = C * torch.exp(-eta / safe_denom + const);
            return torch.where(inside, values, torch.zeros_like(values));

        elif self.test_func_type == "PC-poly":
            assert self.pq is not None;
            p : int = self.pq;
            q : int = self.pq;
            C : float = 1.0 / (p ** p * q ** q) * ((p + q) / (float(b) - float(a))) ** (p + q);
            inside = (t >= float(a)) & (t <= float(b));
            t_a    = torch.clamp(t - float(a), min = 0.0);
            b_t    = torch.clamp(float(b) - t, min = 0.0);
            values = C * (t_a ** p) * (b_t ** q);
            return torch.where(inside, values, torch.zeros_like(values));

        else:
            raise ValueError("Unsupported weak-form test function type: %s" % str(self.test_func_type));



    def add_weight_functions(self,
                             params_row : numpy.ndarray | torch.Tensor | list | tuple,
                             timesteps  : torch.Tensor) -> None:
        r"""
        Build and store weak-form weight functions for one parameter value.

        This method appends/replaces the entries for `params_row` in
        `weight_function_derivatives` without clearing any other parameter values. The `k`'th
        derivative tensor is stored in `weight_function_derivatives[k][param_key]` and has shape
        `(n_weight_functions, n_t)`.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        params_row : numpy.ndarray or torch.Tensor or list or tuple
            The parameter values associated with this time grid. These values are converted into a
            dictionary key using `_param_key(...)`.

        timesteps : torch.Tensor, shape = (n_t,)
            One-dimensional time grid on which the weight functions and their derivatives should be
            evaluated.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Checks
        assert self.type == "weak", "add_weight_functions is only valid for weak LatentDynamics objects";
        assert isinstance(timesteps, torch.Tensor), "timesteps must be a torch.Tensor";
        assert timesteps.ndim == 1, "timesteps must be a 1D tensor";
        assert timesteps.shape[0] > 1, "timesteps must contain at least two time values";
        assert self.test_func_width is not None and self.overlap is not None;

        # Get weight function supports
        key : tuple[float, ...] = self._param_key(params_row);
        L   : float = float(self.test_func_width);
        s   : float = L * float(self.overlap);
        T   : float = float(timesteps[-1].detach().cpu().item());
        a_s, b_s = self._get_support_intervals(T = T, L = L, s = s);
        
        # Determine number of weight functions, time values.
        n_weight_function   : int = len(a_s);
        n_t                 : int = int(timesteps.shape[0]);
        LOGGER.info("Number of %s weak-form weight functions: %d" % (str(self.test_func_type), n_weight_function));

        # Evaluate the weight functions and its derivatives on the time grid.
        derivative_rows : list[list[torch.Tensor]] = [[] for _ in range(self.n_IC + 1)];
        base_t          : torch.Tensor             = timesteps.detach().clone().requires_grad_(True);
        for h in range(n_weight_function):
            current : torch.Tensor = self._weak_weight_function(base_t, float(a_s[h]), float(b_s[h]));
            derivative_rows[0].append(current.detach());
            for k in range(1, self.n_IC + 1):
                grad_outputs = torch.ones_like(current);
                current = torch.autograd.grad(outputs        = current,
                                              inputs         = base_t,
                                              grad_outputs   = grad_outputs,
                                              create_graph   = (k < self.n_IC),
                                              retain_graph   = True)[0];
                derivative_rows[k].append(current.detach());

        for k in range(self.n_IC + 1):
            tensor_k = torch.stack(derivative_rows[k], dim = 0).reshape(n_weight_function, n_t);
            self.weight_function_derivatives[k][key] = tensor_k;

        return;



    def get_test_functions(self,
                           params_row : numpy.ndarray | torch.Tensor | list | tuple) -> list[torch.Tensor]:
        r"""
        Return stored weak-form weight functions for one parameter value.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        params_row : numpy.ndarray or torch.Tensor or list or tuple
            The parameter values whose weak-form weight functions should be returned.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        weight_function_derivatives : list[torch.Tensor]
            A list of length `self.n_IC + 1`. Entry `k` is a tensor of shape
            `(n_weight_functions, n_t)` holding the `k`'th time derivatives of the weight
            functions for `params_row`.
        """

        assert self.type == "weak", "get_test_functions is only valid for weak LatentDynamics objects";
        key : tuple[float, ...] = self._param_key(params_row);
        outputs : list[torch.Tensor] = [];
        for k in range(self.n_IC + 1):
            if key not in self.weight_function_derivatives[k]:
                raise KeyError("No weak-form weight functions found for params=%s (key=%s), derivative order %d" % (
                    str(params_row), str(key), k));
            outputs.append(self.weight_function_derivatives[k][key]);

        shapes = [tuple(tensor.shape) for tensor in outputs];
        assert len(set(shapes)) == 1, "Stored weak-form derivative tensors must have matching shapes";
        return outputs;



    def get_train_coefs(self, params_row : numpy.ndarray | torch.Tensor | list | tuple) -> dict[str, torch.Tensor]:
        r"""
        Fetch the native coefficient dictionary for one parameter combination.

        This method deliberately performs a direct dictionary lookup using `_param_key(...)`. If the
        requested parameter is missing, Python raises a KeyError. This is intentional: all training
        coefficients should be initialized before training starts.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        params_row : numpy.ndarray or torch.Tensor or list or tuple
            The parameter values whose coefficient dictionary we want to fetch.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        coefs : dict[str, torch.Tensor]
            A native coefficient dictionary for the requested parameter. The exact keys depend on
            the concrete LatentDynamics subclass. For example, SINDy uses `A` and `b`, while the
            damped-spring models use `K`, `C`, and `b`.
        """

        key = self._param_key(params_row);
        return self.train_coefs[key];



    def set_train_coefs(self, params_row : numpy.ndarray | torch.Tensor | list | tuple, coefs : dict[str, torch.Tensor]) -> None:
        r"""
        Store a native coefficient dictionary for one parameter combination.

        The values in `coefs` are converted to detached trainable leaf tensors unless they already
        are trainable leaves. This ensures that `trainable_coef_tensors()` can pass these exact tensor
        objects to a torch optimizer and that optimizer updates modify the coefficients stored in
        `self.train_coefs`.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        params_row : numpy.ndarray or torch.Tensor or list or tuple
            The parameter values associated with the coefficient dictionary. These values are
            converted to a tuple key using `param_key(...)`.

        coefs : dict[str, torch.Tensor]
            Native coefficient dictionary. Keys must be strings and values must be tensors. The
            expected keys are subclass-specific.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        assert isinstance(coefs, dict), "coefs must be a dict[str, torch.Tensor]";
        for name, value in coefs.items():
            assert isinstance(name, str), "coefficient names must be strings";
            assert isinstance(value, torch.Tensor), "coefficient %s must be a torch.Tensor" % name;
            if value.requires_grad and value.is_leaf:
                coefs[name] = value;
            else:
                coefs[name] = value.detach().clone().requires_grad_(True);
        self.train_coefs[self._param_key(params_row)] = coefs;
        return;



    def trainable_coef_tensors(self) -> list[torch.Tensor]:
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

        raise RuntimeError("Abstract function LatentDynamics.trainable_coef_tensors!");



    def flatten_coefficients(self, coefs : dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]) -> numpy.ndarray:
        r"""
        Flatten native coefficient dictionaries into one coefficient matrix.

        This is used only at compatibility boundaries, such as coefficient heatmap output. Internal
        training and simulation should use native dictionaries.

        The base implementation does not assume any subclass-specific coefficient names or legacy
        ordering. Instead, for each coefficient dictionary, it loops through the dictionary items in
        insertion order, flattens each tensor, and concatenates those flattened arrays into one row.
        Subclasses may override this method only if they need a specific scalar ordering.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        coefs : dict[str, torch.Tensor] or list[dict[str, torch.Tensor]]
            One native coefficient dictionary or a list of native coefficient dictionaries.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        coef_matrix : numpy.ndarray, shape = (n_param, n_coefs)
            Flattened coefficient matrix. The i'th row contains all flattened coefficient tensors
            from the i'th native coefficient dictionary, concatenated in dictionary item order.
        """

        # Normalize the input so the single-coefficient and multi-coefficient cases share the same
        # validation/flattening logic.
        coefs_list = [coefs] if isinstance(coefs, dict) else coefs;
        assert isinstance(coefs_list, list), "coefs must be a dict or a list of dicts";
        assert len(coefs_list) > 0, "coefs must be non-empty";

        rows : list[numpy.ndarray] = [];
        for coef_dict in coefs_list:
            assert isinstance(coef_dict, dict), "Each coefficient set must be a dictionary";
            assert len(coef_dict) > 0, "Coefficient dictionaries must be non-empty";

            parts : list[numpy.ndarray] = [];
            for name, tensor in coef_dict.items():
                assert isinstance(name, str), "Coefficient names must be strings";
                assert isinstance(tensor, torch.Tensor), "Coefficient %s must be a torch.Tensor" % name;
                parts.append(tensor.detach().cpu().numpy().reshape(-1));

            rows.append(numpy.concatenate(parts, axis = 0).reshape(1, -1));

        return numpy.concatenate(rows, axis = 0);



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
                    params          : numpy.ndarray | None  = None) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
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

        params : numpy.ndarray, shape = (n_param, n_p), optional
            The i'th row holds the i'th combination of parameter values. This can be used by latent 
            dynamics models that depend explicitly on parameter values (e.g., for time-varying or 
            parameterized forcing). Default is None for latent dynamics that don't use parameters.

        
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        loss_LD, loss_coef, loss_stab. 

        loss_LD : list[torch.Tensor], len = n_param
            The i'th element of this list is a 0-dimensional tensor whose lone element holds the 
            sum of the latent dynamics losses from the i'th combination of parameter values. 

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
        r"""
        Export latent-dynamics metadata and LD-owned training coefficients.

        Coefficients are detached and moved to CPU for portable checkpoint/restart files. Loading
        re-creates trainable leaf tensors, so optimizer construction after load still works.
        """

        train_coefs_cpu : dict[tuple[float, ...], dict[str, torch.Tensor]] = {};
        for key, coef_dict in self.train_coefs.items():
            assert isinstance(coef_dict, dict), "train_coefs values must be dictionaries";
            train_coefs_cpu[key] = {};
            for name, tensor in coef_dict.items():
                assert isinstance(name, str);
                assert isinstance(tensor, torch.Tensor);
                train_coefs_cpu[key][name] = tensor.detach().cpu().clone();

        param_dict = {'n_z'             : self.n_z, 
                      'n_coefs'         : self.n_coefs, 
                      'n_IC'            : self.n_IC,
                      'type'            : self.type,
                      'config'          : self.config,
                      'Uniform_t_Grid'  : self.Uniform_t_Grid,
                      'train_coefs'     : train_coefs_cpu};
        return param_dict;



    def load(self, dict_ : dict) -> None:
        r"""
        Load latent-dynamics metadata and replace `self.train_coefs`.

        Shape/model metadata are checked against the already-constructed object. Coefficients are
        restored as trainable leaf tensors rather than raw checkpoint tensors.
        """

        assert(self.n_z             == dict_['n_z']);
        assert(self.n_coefs         == dict_['n_coefs']);
        assert(self.n_IC            == dict_['n_IC']);
        assert(self.type            == dict_.get('type', self.type));
        assert(self.Uniform_t_Grid  == dict_['Uniform_t_Grid']);

        loaded_train_coefs = dict_.get('train_coefs', {});
        assert isinstance(loaded_train_coefs, dict), "train_coefs must be a dictionary";
        self.train_coefs = {};
        for key, coef_dict in loaded_train_coefs.items():
            assert isinstance(key, tuple), "train_coefs keys must be parameter tuples";
            assert isinstance(coef_dict, dict), "train_coefs values must be dictionaries";
            self.train_coefs[key] = {};
            for name, tensor in coef_dict.items():
                assert isinstance(name, str), "coefficient names must be strings";
                assert isinstance(tensor, torch.Tensor), "coefficient values must be tensors";
                self.train_coefs[key][name] = tensor.detach().clone().requires_grad_(True);
        return;
    
