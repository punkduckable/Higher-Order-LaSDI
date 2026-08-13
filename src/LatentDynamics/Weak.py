# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  logging;

import  numpy;
import  torch;

from    LatentDynamics.LatentDynamics   import LatentDynamics;

# Logger setup.
LOGGER : logging.Logger = logging.getLogger(__name__);


# -------------------------------------------------------------------------------------------------
# Weak LatentDynamics class
# -------------------------------------------------------------------------------------------------

class WeakLatentDynamics(LatentDynamics):
    """
    A sub-class of LatentDynamics that enforces its latent dynamics using weak forms.


    -----------------------------------------------------------------------------------------------
    Class/instance variables
    -----------------------------------------------------------------------------------------------

    weight_function_derivatives : list[dict[tuple[float, ...], torch.Tensor]]
        Weak-form weight-function derivative tensors indexed first by derivative order and then by
        parameter tuple. Entry `k` stores the `k`'th time derivatives of the weight functions.
    """
    weight_function_derivatives : list[dict[tuple[float, ...], torch.Tensor]];


    def __init__(   self,
                    n_z             : int,
                    n_coefs         : int,
                    n_IC            : int,
                    Uniform_t_Grid  : bool,
                    trainable       : bool,
                    config          : dict) -> None:
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

        trainable : bool
            Indicates if the trainer should train the latent dynamics parameters. If false,
            `trainable_coef_tensors` should return an empty list.

        config : dict
            The "latent_dynamics" sub-dictionary of the config file. If `type == "weak"`, the
            model-specific sub-dictionary `config[config["type"]]` must contain `overlap`,
            `test_func_width`, and `test_func_type`.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Call base-class initializer directly. Several concrete weak classes explicitly call both
        # InterpolatableLatentDynamics.__init__ and WeakLatentDynamics.__init__, so we avoid
        # cooperative MRO here.
        LatentDynamics.__init__(self,
                                n_z                = n_z,
                                n_coefs            = n_coefs,
                                n_IC               = n_IC,
                                Uniform_t_Grid     = Uniform_t_Grid,
                                trainable          = trainable,
                                stochastic         = getattr(self, "stochastic", False),
                                config             = config)

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
        self.pq              = n_IC + 2;
        self.weight_function_derivatives = [{} for _ in range(self.n_IC + 1)];



    # ---------------------------------------------------------------------------------------------
    # Weak-form specific helpers.
    # ---------------------------------------------------------------------------------------------

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
            a_h : float = float(a_s[h]);
            b_h : float = float(b_s[h]);

            current : torch.Tensor = self._weak_weight_function(base_t, a_h, b_h);
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
