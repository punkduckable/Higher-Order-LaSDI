# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  numpy;
from    sklearn.gaussian_process.kernels    import  ConstantKernel, RBF;
from    sklearn.gaussian_process            import  GaussianProcessRegressor;




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

    X: A 2d numpy array of shape (n_train, input_dim), where n_train is the number of training 
    examples and input_dim is the dimension of the input space to the GPs.

    Y: A 2d numpy array of shape (n_train, n_GPs), where n_train is the number of training 
    examples and n_GPs is the number of GPs.

    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    A list of trained GP regressor objects. If Y has n_GPs columns, then the returned list has k 
    elements. It's i'th element holds a trained GP regressor object whose training inputs are the 
    columns of X and whose corresponding target values are the elements of the i'th column of Y.
    """

    # Checks.
    assert(isinstance(Y, numpy.ndarray));
    assert(isinstance(X, numpy.ndarray));
    assert(len(Y.shape)         == 2);
    assert(len(X.shape)         == 2);
    assert(X.shape[0]           == Y.shape[0]);

    # Setup.
    n_GPs   : int   = Y.shape[1];

    # Transpose Y so that each row corresponds to a particular coefficient. This allows us to 
    # iterate over the GPs by iterating through the rows of Y.
    Y = Y.T;

    # Initialize a list to hold the trained GP objects.
    gp_list : list[GaussianProcessRegressor] = [];

    # Fit the GPs
    for i in range(n_GPs):
        targets_i   : numpy.ndarray     = Y[i, :];

        # Make the kernel.
        # kernel = ConstantKernel() * Matern(length_scale_bounds = (0.01, 1e5), nu = 1.5)
        kernel  = ConstantKernel() * RBF(length_scale_bounds = (0.1, 1e5));

        # Initialize the GP object.
        gp      = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 10, random_state = 1);

        # Fit it to the data (train), then add it to the list of trained GPs
        gp.fit(X, targets_i);
        gp_list += [gp];

    # All done!
    return gp_list;



def eval_gp(gp_list : list[GaussianProcessRegressor], Inputs : numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
    Computes the mean and std of each GP's posterior distribution when evaluated at each 
    combination of parameter values in Inputs.


    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    gp_list: a n_GPs element list of trained GP regressor objects. The i'th element of this list 
    is a GP regressor object that predicts the i'th coefficient. 
    
    Inputs: A 2d numpy.ndarray object of shape (n_inputs, input_dim), where input_dim is the 
    dimensionality of the input space for the GPs) and n_inputs is the number of inputs at which 
    we want to evaluate the posterior distribution of the the GPs. The i,j element of this array 
    specifies the value of the j'th parameter in the i'th combination of parameters. We use this 
    as the testing set for the GP evaluation.


    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------  

    A two element tuple: M and SD. Both are 2d numpy arrays of shape (n_inputs, n_GPs). They
    hold the predicted means and std's for each parameter at each training example, respectively. 
    
    Thus, the i,j element of the M holds the predicted mean of the j'th GP's posterior distribution
    at the i'th input. Likewise, the i,j element of SD holds the standard deviation of the posterior
    distribution for the j'th GP evaluated at the i'th input.
    """

    # Checks
    assert(isinstance(gp_list, list));
    assert(isinstance(Inputs, numpy.ndarray));
    assert(len(Inputs.shape) == 2);

    # Setup 
    n_GPs       : int   = len(gp_list);
    n_inputs    : int   = Inputs.shape[0];
    pred_mean   : numpy.ndarray     = numpy.zeros([n_inputs, n_GPs]);
    pred_std    : numpy.ndarray     = numpy.zeros([n_inputs, n_GPs]);

    # Find the means and SDs of the posterior distribution for each GP evaluated at the 
    # various inputs.
    for i in range(n_GPs):
        GP_i : GaussianProcessRegressor = gp_list[i];
        pred_mean[:, i], pred_std[:, i] = GP_i.predict(Inputs, return_std = True);

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

    gp_list: a n_GPs element list of trained GP regressor objects.

    Input: An numpy.ndarray array with shape (input_dim) that holds a single combination of 
    parameter values. i.e., a single test example. Here, input_dim is the dimension of the input 
    space for the GPs. We evaluate the posterior distribution of each GP in gp_list at this input 
    (getting a prediction for each GP).

    n_samples: Number of samples we draw from each GP's posterior distribution. 
    

    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    A 2d numpy ndarray object, coef_samples, with shape (n_samples, n_GPs) whose i,j element holds 
    the i'th sample of the posterior distribution for the j'th GP evaluated at the Input.
    """

    # Checks.
    assert(isinstance(gp_list, list));
    assert(isinstance(Input, numpy.ndarray));
    assert(len(Input.shape) == 1);

    # Setup.
    n_GPs           : int           = len(gp_list);
    coef_samples    : numpy.ndarray = numpy.zeros([n_samples, n_GPs]);

    # Evaluate the predicted mean and std at the Input.
    pred_mean, pred_std = eval_gp(gp_list, Input.reshape(1, -1));
    pred_mean   = pred_mean[0];
    pred_std    = pred_std[0];

    # Cycle through the samples and coefficients. For each sample of the k'th coefficient, we draw
    # a sample from the normal distribution with mean pred_mean[k] and std pred_std[k].
    for s in range(n_samples):
        for k in range(n_GPs):
            coef_samples[s, k] = numpy.random.normal(pred_mean[k], pred_std[k]);

    # All done!
    return coef_samples;