# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  torch;
import  numpy;
import  warnings;
from    sklearn.gaussian_process.kernels    import  ConstantKernel, RBF, Matern;
from    sklearn.gaussian_process            import  GaussianProcessRegressor;
from    sklearn.exceptions                  import  ConvergenceWarning;

# Set up logging.
import  logging;
LOGGER = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# GP Interpolate class
# -------------------------------------------------------------------------------------------------

class GPInterpolate:
    r"""
    GP-backed interpolation for `Interpolatable` latent-dynamics objects.

    This class fits one independent Gaussian process for each scalar component of each named
    coefficient tensor. The public methods return coefficient dictionaries with the same native
    keys and tensor shapes as the latent-dynamics model's `train_coefs` entries.
    """

    def __init__(self, train_coefs : dict[tuple[float, ...], dict[str, torch.Tensor]]) -> None:
        r"""
        Build one collection of GPs for each named coefficient tensor.

        For a fixed tensor name (for example "A" or "K"), every training parameter must have a
        tensor with the same shape. We flatten that tensor component-wise and fit one independent GP
        per scalar component, using the parameter tuple as GP input.
        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        train_coefs : dict[tuple[float, ...], dict[str, torch.Tensor]]
            LD-owned training coefficient dictionary. The outer key is an exact parameter tuple;
            the inner dictionary maps coefficient tensor names to tensors.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        
        """

        # Set the training coefficients.
        self.update_train_coefs(train_coefs)


    @staticmethod
    def _param_array(param : numpy.ndarray | torch.Tensor | list | tuple) -> numpy.ndarray:
        r"""
        Normalize a parameter input to a one-dimensional NumPy array for GP evaluation.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param : numpy.ndarray or torch.Tensor or list or tuple
            Parameter values for one requested point.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        param_np : numpy.ndarray, shape = (n_p,)
            One-dimensional float64 NumPy array containing the parameter values.
        """

        if isinstance(param, torch.Tensor):
            param = param.detach().cpu().numpy();
        elif isinstance(param, (list, tuple)):
            param = numpy.array(param);
        assert isinstance(param, numpy.ndarray), "param must be numpy.ndarray, torch.Tensor, list, or tuple";
        return param.reshape(-1).astype(numpy.float64);



    def update_train_coefs(self, train_coefs : dict[tuple[float, ...], dict[str, torch.Tensor]]) -> None:
        """
        This method updates self's train_coefs attribute use the passed dictionary.

        Specifically, this method builds one collection of GPs for each named coefficient tensor.

        For a fixed tensor name (for example "A" or "K"), every training parameter must have a
        tensor with the same shape. We flatten that tensor component-wise and fit one independent GP
        per scalar component, using the parameter tuple as GP input.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        train_coefs : dict[tuple[float, ...], dict[str, torch.Tensor]]
            LD-owned training coefficient dictionary. The outer key is an exact parameter tuple;
            the inner dictionary maps coefficient tensor names to tensors.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Check the top-level object first. The class is intentionally narrow for now: it only
        # supports LD train_coefs dictionaries with tensor-valued inner dictionaries.
        assert isinstance(train_coefs, dict), "train_coefs must be a dictionary";
        assert len(train_coefs) > 0, "train_coefs must be non-empty";

        # Store parameter keys in a deterministic list. This order is used to build both the GP
        # input array X and the corresponding target rows for every coefficient tensor.
        self.train_coefs = train_coefs;
        self.param_keys : list[tuple[float, ...]] = list(train_coefs.keys());
        for key in self.param_keys:
            assert isinstance(key, tuple), "train_coefs keys must be parameter tuples";
            assert all(isinstance(x, float) for x in key), "parameter tuple entries must be floats";

        # Use the first coefficient dictionary as the schema, then verify that every other
        # parameter has exactly the same names and shapes.
        first = train_coefs[self.param_keys[0]];
        assert isinstance(first, dict), "train_coefs values must be dictionaries";
        self.coef_names : list[str] = list(first.keys());
        assert len(self.coef_names) > 0, "coefficient dictionaries must be non-empty";
        for name in self.coef_names:
            assert isinstance(name, str), "coefficient names must be strings";
            assert isinstance(first[name], torch.Tensor), "coefficient values must be tensors";

        self.coef_shapes : dict[str, torch.Size] = {name: first[name].shape for name in self.coef_names};
        for key, coef_dict in train_coefs.items():
            assert isinstance(coef_dict, dict), "train_coefs[%s] must be a dictionary" % str(key);
            assert set(coef_dict.keys()) == set(self.coef_names), "coefficient keys differ for parameter %s" % str(key);
            for name in self.coef_names:
                assert isinstance(coef_dict[name], torch.Tensor), "coefficient %s for parameter %s must be a tensor" % (name, str(key));
                assert coef_dict[name].shape == self.coef_shapes[name], "coefficient %s shape mismatch for parameter %s" % (name, str(key));

        # GP inputs: one row per training parameter.
        self.X : numpy.ndarray = numpy.array(self.param_keys, dtype = numpy.float64);
        assert len(self.X.shape) == 2, "parameter keys must form a 2D array";

        # For each named tensor, flatten it across components and fit one GP per component. The
        # existing GaussianProcess utilities handle scaling, kernel construction, and sampling.
        self.gps : dict[str, list[GaussianProcessRegressor]] = {};
        for name in self.coef_names:
            Y_rows : list[numpy.ndarray] = [];
            for key in self.param_keys:
                Y_rows.append(train_coefs[key][name].detach().cpu().numpy().reshape(1, -1));
            Y : numpy.ndarray = numpy.concatenate(Y_rows, axis = 0);
            self.gps[name] = fit_gps(self.X, Y);
            LOGGER.info("Fit %d GPs for coefficient tensor '%s' with shape %s" % (Y.shape[1], name, tuple(self.coef_shapes[name])));
        return;


    def sample(self, param : numpy.ndarray | torch.Tensor | list | tuple) -> dict[str, torch.Tensor]:
        r"""
        Draw one sample from the posterior distributions for each coefficient, when these 
        distributions are conditioned on the passed parameter value. 
        
        The returned dictionary has the same keys and tensor shapes as each item in `train_coefs`,
        so it can be passed directly to `LatentDynamics.simulate(...)`.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param : numpy.ndarray or torch.Tensor or list or tuple
            Parameter values at which to sample the coefficient posterior.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        out : dict[str, torch.Tensor]
            Native coefficient dictionary containing one posterior sample for each coefficient
            tensor.
        """

        x = self._param_array(param);
        out : dict[str, torch.Tensor] = {};
        for name in self.coef_names:
            sample_np : numpy.ndarray = sample_coefs(self.gps[name], x, 1)[0, :].reshape(tuple(self.coef_shapes[name]));
            out[name] = torch.tensor(sample_np, dtype = torch.float32);
        return out;



    def mean(self, param : numpy.ndarray | torch.Tensor | list | tuple) -> dict[str, torch.Tensor]:
        r"""
        Return the posterior mean coefficient dictionary at a requested parameter value.

        This is used for deterministic plotting/rollouts where drawing random coefficient samples
        would make figures nondeterministic.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param : numpy.ndarray or torch.Tensor or list or tuple
            Parameter values at which to evaluate the posterior mean.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        out : dict[str, torch.Tensor]
            Native coefficient dictionary containing the posterior mean for each coefficient
            tensor.
        """

        x = self._param_array(param);
        out : dict[str, torch.Tensor] = {};
        for name in self.coef_names:
            mean_np, _ = eval_gp(self.gps[name], x.reshape(1, -1));
            out[name] = torch.tensor(mean_np[0, :].reshape(tuple(self.coef_shapes[name])), dtype = torch.float32);
        return out;



    def std(self, param : numpy.ndarray | torch.Tensor | list | tuple) -> dict[str, torch.Tensor]:
        r"""
        Return the standard-deviation of the posterior distributions for each coefficient 
        conditioned on requested parameter.

        The returned tensors use the same native keys and shapes as coefficients in the coefficient 
        dictionary. Each entry holds the marginal GP posterior standard deviation for the 
        corresponding scalar coefficient.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param : numpy.ndarray or torch.Tensor or list or tuple
            Parameter values at which to evaluate posterior standard deviations.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        out : dict[str, torch.Tensor]
            Native coefficient dictionary containing posterior standard deviations for each
            coefficient tensor.
        """

        x = self._param_array(param);
        out : dict[str, torch.Tensor] = {};
        for name in self.coef_names:
            _, std_np = eval_gp(self.gps[name], x.reshape(1, -1));
            out[name] = torch.tensor(std_np[0, :].reshape(tuple(self.coef_shapes[name])), dtype = torch.float32);
        return out;



# -------------------------------------------------------------------------------------------------
# Gaussian Process functions! 
# -------------------------------------------------------------------------------------------------

def fit_gps(X : numpy.ndarray, Y : numpy.ndarray) -> list[GaussianProcessRegressor]:
    r"""
    Trains a GP for each column of Y. If Y has shape n_train x n_GPs, then we train k GP 
    regressors. In this case, we assume that X has shape n_train x input_dim. Thus, the Input to 
    the GP is in \mathbb{R}^input_dim. For each k, we train a GP where the i'th row of X is the 
    input and the i,k component of Y is the corresponding target. We assume the target coefficients 
    are independent.
    
    We return a list of n_GPs GP Regressor objects, the k'th one of which makes predictions for 
    the k'th coefficient in the latent dynamics. 


    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    X : numpy.ndarray, shape = (n_train, input_dim) 
        For each column of Y, we treat the rows of X and entry of the column of Y as samples of 
        the input and target random variables, respectively. We fit a GP on this data. Thus, 
        n_train is the number of training examples and input_dim is the dimension of the input 
        space to the GPs. 

    Y : numpy.ndarray, shape = (n_train, n_GPs)
        For each column of Y, we treat the rows of X and entry of the column of Y as samples of 
        the input and target random variables, respectively. We fit a GP on this data. Thus, 
        n_train is the number of training examples and input_dim is the dimension of the input 
        space to the GPs. 
    
    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    gp_list : list[GaussianProcessRegressor], len = n_GPs
        The j'th element holds a trained GP regressor object whose training inputs are the 
        rows of X and whose corresponding target values are the elements of the j'th column of Y.
    """

    # Checks.
    assert isinstance(Y, numpy.ndarray),        "type(Y) = %s" % str(type(Y));
    assert isinstance(X, numpy.ndarray),        "type(X) = %s" % str(type(X));
    assert len(Y.shape)         == 2,           "Y.shape = %s" % str(Y.shape);
    assert len(X.shape)         == 2,           "X.shape = %s" % str(X.shape);
    assert X.shape[0]           == Y.shape[0],  "X.shape = %s, Y.shape = %s" % (str(X.shape), str(Y.shape));

    # Setup.
    n_GPs       : int   = Y.shape[1];
    n_inputs    : int   = X.shape[1];

    # Scale inputs to improve conditioning of kernel hyperparameter optimization.
    # This is especially important when parameters have very different magnitudes
    # (e.g., ~1e-9 and ~1e-4), which can trigger many ConvergenceWarnings.
    x_mean  : numpy.ndarray = numpy.mean(X, axis = 0);
    x_std   : numpy.ndarray = numpy.std(X, axis = 0, ddof = 1);  # Use unbiased estimator
    LOGGER.info(f"Input scaling: X_mean = {x_mean}, X_std = {x_std}");
    
    # Protect against near-zero / non-finite std in any input dimension.
    #
    # IMPORTANT: We must be careful with *absolute* thresholds here because some parameters
    # (e.g., laser power) can live at O(1e-9) but still be meaningfully varying. We therefore
    # use a scale-aware threshold based on the observed range of the parameter values.
    #
    # Also note: numpy.std(..., ddof=1) returns NaN when n_train < 2. We explicitly guard
    # against this so we don't propagate NaNs into Xs.
    for idx in range(n_inputs):
        rng_idx : float = float(X[:, idx].max() - X[:, idx].min());
        eps     : float = float(max(1e-15, 1e-6 * rng_idx));

        # If std is NaN/Inf (e.g., n_train < 2) or extremely small relative to the range,
        # replace it with something safe.
        if (not numpy.isfinite(x_std[idx])) or (x_std[idx] < eps):
            if rng_idx <= 0.0 or (not numpy.isfinite(rng_idx)):
                # Truly constant dimension -> ignore it by using x_std = 1.0 (scaled values ~ 0).
                LOGGER.warning(f"Input dimension {idx}: non-finite/near-zero x_std={x_std[idx]:.2e} with rng={rng_idx:.2e}. Treating as constant (x_std=1.0).");
                x_std[idx] = 1.0;
            else:
                # Dimension varies, but std is too small / non-finite -> clamp to eps to preserve scale.
                LOGGER.warning(f"Input dimension {idx}: non-finite/near-zero x_std={x_std[idx]:.2e} relative to rng={rng_idx:.2e}. Clamping x_std to eps={eps:.2e}.");
                x_std[idx] = eps;

    Xs: numpy.ndarray = (X - x_mean) / x_std;

    # Initialize a list to hold the trained GP objects.
    gp_list : list[GaussianProcessRegressor] = [];

    # Fit the GPs
    for i in range(n_GPs):
        # Fetch the i'th column of Y (target values for the i'th GP).
        targets_i   : numpy.ndarray     = Y[:, i];

        # Scale targets per coefficient (each GP has its own target distribution).
        ith_mean: float = float(numpy.mean(targets_i));
        ith_std: float  = float(numpy.std(targets_i, ddof = 1));  # Use unbiased estimator

        # Protect against non-finite / near-zero target std (e.g., ddof=1 with n_train < 2).
        # Use a scale-aware threshold based on the observed range of the targets.
        ith_rng: float  = float(numpy.max(targets_i) - numpy.min(targets_i));
        ith_eps: float  = float(max(1e-15, 1e-6 * ith_rng));
        LOGGER.debug(f"GP {i}: ith_mean = {ith_mean:.6e}, ith_std = {ith_std:.6e}, targets_i range = [{numpy.min(targets_i):.6e}, {numpy.max(targets_i):.6e}]");
        if (not numpy.isfinite(ith_std)) or (ith_std < ith_eps):
            if ith_rng <= 0.0 or (not numpy.isfinite(ith_rng)):
                LOGGER.warning(f"GP coefficient {i}: non-finite/near-zero ith_std={ith_std:.2e} with rng={ith_rng:.2e}. Treating as constant (ith_std=1.0).");
                ith_std = 1.0;
            else:
                LOGGER.warning(f"GP coefficient {i}: non-finite/near-zero ith_std={ith_std:.2e} relative to rng={ith_rng:.2e}. Clamping ith_std to eps={ith_eps:.2e}.");
                ith_std = ith_eps;
        targets_i_s: numpy.ndarray = (targets_i - ith_mean) / ith_std;

        # Make the kernel.
        # Option 1: Matern kernel (recommended for smooth but non-infinitely-differentiable functions)
        # Length scales tuned for normalized parameter space (mean = 0, std = 1).
        # For a 5x5 grid, typical distances are O(1), so length scales of 0.5-5 give smooth interpolation.
        # Increased minimum to 0.5 to prevent overfitting to local patterns.
        kernel  = ConstantKernel(constant_value = 1.0, constant_value_bounds = (1e-3, 1e3)) * \
                  Matern(length_scale = 1.0, length_scale_bounds = (1.0, 1e3), nu = 2.5);
        # Option 2: RBF kernel (for infinitely smooth functions)
        # kernel  = ConstantKernel(constant_value = 1.0, constant_value_bounds = (1e-3, 1e3)) * \
        #           RBF(length_scale_bounds = (0.1, 10.0));

        # Initialize the GP object.
        #
        # alpha: Adds noise to the diagonal of the kernel matrix (observation noise).
        #        Larger values = more uncertainty = less overfitting to training data.
        #        Typical range: 1e-10 (very confident) to 1e-3 (high uncertainty).
        #        Using 4e-4 for tighter, more stable predictions to prevent divergent dynamics.
        #
        # n_restarts_optimizer: Number of random restarts for hyperparameter optimization.
        #                       More restarts = better hyperparameters but slower.
        #                       Using 10 restarts for better kernel tuning and stability.
        ith_gp      = GaussianProcessRegressor(
                            kernel                  = kernel, 
                            alpha                   = 4e-4,     # Tighter uncertainty to prevent divergent coefficients
                            n_restarts_optimizer    = 10,       # More restarts for better hyperparameters
                            random_state            = 1);

        # Fit it to the data (train).
        with warnings.catch_warnings():
            # This warning is common (length_scale near bound) and can print hundreds of times
            # across many coefficients/restarts. It is not fatal, so silence it.
            warnings.filterwarnings("ignore", category = ConvergenceWarning);
            ith_gp.fit(Xs, targets_i_s);

        # Attach scaling so eval_gp/sample_coefs can use physical units.
        ith_gp._x_mean = x_mean;
        ith_gp._x_std  = x_std;
        ith_gp._y_mean = ith_mean;
        ith_gp._y_std  = ith_std;
        
        # Log GP hyperparameters for first few coefficients and every 50th thereafter
        if i < 3 or i % 50 == 0:
            LOGGER.info("GP %d: kernel = %s, log_marginal_likelihood = %.3f" % (
                i, str(ith_gp.kernel_), ith_gp.log_marginal_likelihood_value_));
        
        # Add the trained GP to the list.
        gp_list.append(ith_gp);

    # Log summary statistics across all GPs
    length_scales = [];
    for gp in gp_list:
        if hasattr(gp.kernel_, 'k2') and hasattr(gp.kernel_.k2, 'length_scale'):
            length_scales.append(gp.kernel_.k2.length_scale);
    if len(length_scales) > 0:
        length_scales = numpy.array(length_scales);
        LOGGER.info("GP length scales: min = %.3f, median = %.3f, max = %.3f, mean = %.3f" % (
            numpy.min(length_scales), numpy.median(length_scales), 
            numpy.max(length_scales), numpy.mean(length_scales)));
    
    # All done!
    return gp_list;



def eval_gp(gp_list : list[GaussianProcessRegressor], Inputs : numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
    Computes the mean and std of each GP's posterior distribution when evaluated at each 
    combination of parameter values in Inputs.


    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    gp_list : list[GaussianProcessRegressor], len = n_GPs
       a list of trained GP regressor objects. The i'th element of this list is a GP regressor 
       object whose domain includes the rows of Inputs. These GPs should have a few additional 
       attributes: _x_mean, _x_std, _y_mean, _y_std.
    
    Inputs : numpy.ndarray, shape = (n_inputs, input_dim)
        We evaluate each Gaussian Process in gp_list at each row of Inputs. Thus, the i'th row
        represents the i'th input to the Gaussian Processes. Here, input_dim is the dimensionality 
        of the input space for the GPs) and n_inputs is the number of inputs at which we want to 
        evaluate the posterior distribution of the the GPs. 


    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------  

    M, SD 

    M : numpy.ndarray, shape = (n_inputs, n_GPs)
        the i,j element of the M holds the predicted mean of the j'th GP's posterior distribution
        at the i'th row of Inputs.
    
    SD : numpy.ndarray, shape = (n_inputs, n_GPs)
        the i,j element of SD holds the standard deviation of the posterior distribution for the 
        j'th GP evaluated at the i'th row of Inputs.
    """

    # Checks
    assert isinstance(gp_list, list),           "type(gp_list) = %s" % str(type(gp_list));
    assert isinstance(Inputs, numpy.ndarray),   "type(Inputs) = %s" % str(type(Inputs));
    assert len(Inputs.shape) == 2,              "Inputs.shape = %s" % str(Inputs.shape);

    # Setup 
    n_GPs       : int           = len(gp_list);
    n_inputs    : int           = Inputs.shape[0];
    pred_mean   : numpy.ndarray = numpy.zeros([n_inputs, n_GPs]);
    pred_std    : numpy.ndarray = numpy.zeros([n_inputs, n_GPs]);

    # Find the means and SDs of the posterior distribution for each GP evaluated at the
    # various inputs.
    for i in range(n_GPs):
        ith_gp = gp_list[i];
        
        # Scale inputs to match training data normalization.
        if hasattr(ith_gp, "_x_mean") and hasattr(ith_gp, "_x_std"):
            Scaled_Inputs = (Inputs - ith_gp._x_mean) / ith_gp._x_std;
        else:
            # No scaling attached; use inputs as-is (shouldn't happen if fit_gps was used).
            LOGGER.warning(f"GP {i} missing _x_mean/_x_std attributes. Using unscaled inputs.");
            Scaled_Inputs = Inputs; 

        ith_m_scaled, ith_s_scaled = ith_gp.predict(Scaled_Inputs, return_std = True);

        # Undo target scaling to return predictions in physical units.
        if hasattr(ith_gp, "_y_mean") and hasattr(ith_gp, "_y_std"):
            ith_m = ith_m_scaled * ith_gp._y_std + ith_gp._y_mean; 
            ith_s = ith_s_scaled * ith_gp._y_std;
        else:
            # No scaling attached; use predictions as-is (shouldn't happen if fit_gps was used).
            LOGGER.warning(f"GP {i} missing _y_mean/_y_std attributes. Using unscaled predictions.");
            ith_m = ith_m_scaled;
            ith_s = ith_s_scaled;               

        pred_mean[:, i] = ith_m;
        pred_std[:, i]  = ith_s;

    # All done!
    return pred_mean, pred_std;



def sample_coefs(   gp_list     : list[GaussianProcessRegressor], 
                    Input       : numpy.ndarray, 
                    n_samples   : int) -> numpy.ndarray:
    """
    Generates n_samples samples of the posterior distributions of the GPs in gp_list evaluated at
    the input specified by Input. 
    

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    gp_list : list[GaussianProcessRegressor], len n_GPs
         A list of trained GP regressor objects. They should all use the same input space (which 
         contains Input).

    Input : numpy.ndarray, shape = (input_dim)
        holds a single combination of parameter values. i.e., a single test example. Here, 
        input_dim is the dimension of the input space for the GPs. We evaluate the posterior 
        distribution of each GP in gp_list at this input (getting a prediction for each GP).

    n_samples : int
        The number of samples we draw from each GP's posterior distribution. 
    

    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    coef_samples : numpy.ndarray, shape = (n_samples, n_GPs)
        i,j element holds the i'th sample of the posterior distribution for the j'th GP evaluated 
        at the Input.
    """

    # Checks.
    assert isinstance(gp_list, list),           "type(gp_list) = %s" % str(type(gp_list));
    assert isinstance(Input, numpy.ndarray),    "type(Input) = %s" % str(type(Input));
    assert isinstance(n_samples, int),          "type(n_samples) = %s" % str(type(n_samples));
    assert len(Input.shape) == 1,               "Input.shape = %s" % str(Input.shape);

    # Setup.
    n_GPs           : int           = len(gp_list);
    coef_samples    : numpy.ndarray = numpy.zeros([n_samples, n_GPs]);

    # Evaluate the predicted mean and std at the Input.
    pred_mean, pred_std = eval_gp(gp_list, Input.reshape(1, -1));
    pred_mean   = pred_mean[0]; # Before reshape, pred_mean has shape (1, n_GPs). After reshape, it has shape (n_GPs,).
    pred_std    = pred_std[0];

    # Cycle through the samples and coefficients. For each sample of the k'th coefficient, we draw
    # a sample from the normal distribution with mean pred_mean[k] and std pred_std[k]. Note that 
    # we clip the sample to avoid outlying samples that can lead to numerical instability and 
    # divergent latent dynamics. Clipping to +/- 1.5 sigma (87% confidence interval) prioritizes 
    # stability over exploration to prevent divergent predictions.
    for s in range(n_samples):
        for k in range(n_GPs):
            sample = numpy.random.normal(pred_mean[k], pred_std[k]);
            # Clip to +/- 1.5 std to prevent divergent dynamics
            coef_samples[s, k] = numpy.clip(sample, pred_mean[k] - 1.5*pred_std[k], pred_mean[k] + 1.5*pred_std[k]);
    
    # All done!
    return coef_samples;