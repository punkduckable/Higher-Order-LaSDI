# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  numpy;
import  warnings;
from    sklearn.gaussian_process.kernels    import  ConstantKernel, RBF, Matern;
from    sklearn.gaussian_process            import  GaussianProcessRegressor;
from    sklearn.exceptions                  import  ConvergenceWarning;

# Set up logging.
import  logging;
LOGGER = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# Gaussian Process functions! 
# -------------------------------------------------------------------------------------------------

def fit_gps(X : numpy.ndarray, Y : numpy.ndarray) -> list[GaussianProcessRegressor]:
    """
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
    LOGGER.debug(f"Input scaling: X_mean = {x_mean}, X_std = {x_std}");
    
    # Protect against near-zero std in any input dimension
    for idx in range(n_inputs):
        if x_std[idx] < 1e-8:
            LOGGER.warning(f"Input dimension {idx}: x_std = {x_std[idx]:.2e} is near-zero. Using x_std[{idx}] = 1.0.");
            x_std[idx] = 1.0;
    
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
        # Protect against both zero and near-zero std
        LOGGER.debug(f"GP {i}: ith_mean = {ith_mean:.6e}, ith_std = {ith_std:.6e}, targets_i range = [{numpy.min(targets_i):.6e}, {numpy.max(targets_i):.6e}]");
        if ith_std < 1e-8:  # Threshold for numerical safety
            LOGGER.warning(f"GP coefficient {i}: ith_std = {ith_std:.2e} is near-zero. Using ith_std=1.0. This suggests latent dynamics are nearly constant!");
            ith_std = 1.0;
        targets_i_s: numpy.ndarray = (targets_i - ith_mean) / ith_std;

        # Make the kernel.
        # Option 1: Matern kernel
        kernel  = ConstantKernel(constant_value = 1.0, constant_value_bounds = (1e-5, 1e6)) * \
                  Matern(length_scale_bounds = (1.0, 1e3), nu = 1.5);
        # Option 2: RBF kernel
        # kernel  = ConstantKernel(constant_value = 1.0, constant_value_bounds = (1e-5, 1e6)) * \
        #           RBF(length_scale_bounds = (1e-3, 1e2));

        # Initialize the GP object.
        #
        # alpha: Adds noise to the diagonal of the kernel matrix (observation noise).
        #        Larger values = more uncertainty = less overfitting to training data.
        #        Typical range: 1e-10 (very confident) to 1e-3 (high uncertainty).
        #        We use 1e-2 to add more regularization and reduce variance in predictions,
        #        which helps prevent divergent samples in latent dynamics.
        #
        # n_restarts_optimizer: Number of random restarts for hyperparameter optimization.
        #                       More restarts = better hyperparameters but slower.
        #                       Using 10 restarts for better kernel tuning and stability.
        ith_gp      = GaussianProcessRegressor(
                            kernel                  = kernel, 
                            alpha                   = 1e-2,     # Add noise/uncertainty to predictions (increased from 1e-3 for stability)
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
        
        # Add the trained GP to the list.
        gp_list.append(ith_gp);

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
    # a sample from the normal distribution with mean pred_mean[k] and std pred_std[k]. Note that we 
    # clip the sample to be within 1.0 standard deviation of the mean to avoid outlying samples that 
    # can lead to numerical instability and divergent latent dynamics. This conservative clipping 
    # prioritizes stability over exploration.
    for s in range(n_samples):
        for k in range(n_GPs):
            sample = numpy.random.normal(pred_mean[k], pred_std[k]);
            coef_samples[s, k] = numpy.clip(sample, pred_mean[k] - 2.0*pred_std[k], pred_mean[k] + 2.0*pred_std[k]);
    
    # All done!
    return coef_samples;