# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

# Add the main (src) directory to the search path.
import  os, sys;
src_path            : str   = os.path.abspath(os.path.dirname(os.path.dirname(__file__)));
Utilities_path      : str   = os.path.join(src_path, "Utilities");
EncoderDecoder_path : str   = os.path.join(src_path, "EncoderDecoder");
sys.path.append(Utilities_path);
sys.path.append(EncoderDecoder_path);
sys.path.append(src_path);

import  logging;

from    sklearn.gaussian_process    import  GaussianProcessRegressor;
import  torch;
import  numpy;

from    Enums                       import  NextStep;  
from    Trainer                     import  Trainer;
from    GaussianProcess             import  fit_gps, sample_coefs;
from    Autoencoder                 import  Autoencoder;
from    Autoencoder_Pair            import  Autoencoder_Pair;
from    CNN_3D_Autoencoder          import  CNN_3D_Autoencoder;
from    EncoderDecoder              import  EncoderDecoder;


# Setup logger.
LOGGER : logging.Logger = logging.getLogger(__name__);





# -------------------------------------------------------------------------------------------------
# FOM_Variance function
# -------------------------------------------------------------------------------------------------

def FOM_Variance(trainer : Trainer) -> NextStep:
    """
    This function used greedy sampling to find the element of the trainer's testing set (excluding 
    the training set) whose corresponding latent dynamics gives the highest variance FOM time 
    series. 

    How does this work? The latent space coefficients change with parameter values. For each 
    coefficient, we fit a gaussian process whose input is the parameter values. Thus, for each 
    potential parameter value and coefficient, we can find a distribution for that coefficient 
    when we use that parameter value.

    With this in mind, for each combination of parameters in trainer.param_space's test space, 
    we draw a set of samples of the coefficients at that combination of parameter values. For
    each combination, we solve the latent dynamics forward in time (using the sampled set of
    coefficient values to define the latent dynamics). This gives us a time series of latent 
    states. We do this for each sample, for each testing parameter. 

    For each time step and parameter combination, we get a set of latent frames. We map that 
    set to a set of FOM frames and then find the STD of each component of those FOM frames 
    across the samples. This give us a number. We find the corresponding number for each time 
    step and combination of parameter values and then return the parameter combination that 
    gives the biggest number (for some time step). This becomes the new sample point.

    Thus, the sample point is ALWAYS an element of the testing set. 
    

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    trainer : Trainer
        A Trainer object that we use for training. We sample a new training point 
        from this trainer.


    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    NextStep.RunSample : NextStep
        indicates that we have a new sample and need to generate the FOM solution using the 
        corresponding parameter values for the IC/physics. 
    """


    trainer.timer.start("new_sample");
    assert len(trainer.U_Test)             >  0,                                    "len(trainer.U_Test) = %d" % len(trainer.U_Test);
    assert len(trainer.U_Test)             == trainer.param_space.n_test(),         "len(trainer.U_Test) = %d, trainer.param_space.n_test() = %d" % (len(trainer.U_Test), trainer.param_space.n_test());
    assert trainer.best_train_coefs is not None,                                    "best_train_coefs is None (did training run and checkpoint succeed?)";
    assert trainer.best_train_coefs.shape[0]     == trainer.param_space.n_train(),  "trainer.best_train_coefs.shape[0] = %d, trainer.param_space.n_train() = %d" % (trainer.best_train_coefs.shape[0], trainer.param_space.n_train());

    train_coefs : numpy.ndarray = trainer.best_train_coefs;                     # Shape = (n_train, n_coefs).
    LOGGER.info('\n~~~~~~~ Finding New Point ~~~~~~~');

    # Move the encoder_decoder to the cpu (this is where all the GP stuff happens) and load the 
    # encoder_decoder from the last checkpoint. This should be the one that obtained the best loss 
    # so far. Remember that train_coefs should specify the coefficients from that iteration. 
    LOGGER.info("Sampling: Loading encoder_decoder from checkpoint.");
    encoder_decoder : EncoderDecoder    = trainer.encoder_decoder.cpu();
    n_test          : int               = trainer.param_space.n_test();
    n_train         : int               = trainer.param_space.n_train();
    encoder_decoder.load_state_dict(torch.load(trainer.path_checkpoint + '/' + 'checkpoint.pt', map_location = 'cpu'));

    # First, find the candidate parameters. These are the elements of the testing set that 
    # are not already in the training set.
    candidate_parameters    : list[numpy.ndarray]   = [];
    t_Candidates            : list[torch.Tensor]    = [];
    for i in range(n_test):
        ith_Test_param = trainer.param_space.test_space[i, :];
        
        # Check if the i'th testing parameter is in the training set (all close returns True if
        # the two arrays are equal to within a tolerance)
        in_train : bool = False;
        for j in range(n_train):
            if numpy.allclose(trainer.param_space.train_space[j, :], ith_Test_param, rtol = 1e-12, atol = 1e-14):
                in_train = True;
                break;
        
        # If not, add it to the set of candidates
        if(in_train == False):
            candidate_parameters.append(ith_Test_param);
            t_Candidates.append(trainer.t_Test[i]);
    
    # Concatenate the candidates to form an array of shape (n_candidates, n_coefs).
    n_candidates : int = len(candidate_parameters);
    LOGGER.info("There are %d candidate testing parameters (%d in the testing space, %d in the training set)" % (n_candidates, n_test, n_train));
    assert n_candidates >= 1, "n_candidates = %d" % n_candidates;
    candidate_parameters    = numpy.array(candidate_parameters);


    # Map the initial conditions for the FOM to initial conditions in the latent space.
    # Yields an n_candidates element list whose i'th element is an n_IC element list whose j'th
    # element is an numpy.ndarray of shape (1, n_z) whose k'th element holds the k'th component
    # of the encoding of the initial condition for the j'th derivative of the latent dynamics 
    # corresponding to the i'th candidate combination of parameter values.
    Z0 : list[list[numpy.ndarray]]  = encoder_decoder.latent_initial_conditions(  
                                                                param_grid  = candidate_parameters, 
                                                                physics     = trainer.physics,
                                                                trainer     = trainer);

    # Log coefficient statistics before fitting GPs (this is critical for debugging!)
    LOGGER.info("Coefficient statistics for GP fitting:");
    LOGGER.info("  Training parameters shape: %s" % str(trainer.param_space.train_space.shape));
    LOGGER.info("  Coefficients shape: %s" % str(train_coefs.shape));
    for coef_idx in range(min(5, train_coefs.shape[1])):  # Log first 5 coefficients
        coef_vals = train_coefs[:, coef_idx];
        LOGGER.info("  Coef %d: mean=%.6e, std=%.6e, min=%.6e, max=%.6e, range=%.6e" % (
            coef_idx, numpy.mean(coef_vals), numpy.std(coef_vals), 
            numpy.min(coef_vals), numpy.max(coef_vals), numpy.max(coef_vals) - numpy.min(coef_vals)));
    
    # Train the GPs on the training data, get one GP per latent space coefficient.
    gp_list : list[GaussianProcessRegressor] = fit_gps(trainer.param_space.train_space, train_coefs);

    # For each combination of parameter values in the candidate set, for each coefficient, 
    # draw a set of samples from the posterior distribution for that coefficient evaluated at
    # the candidate parameters. We store the samples for a particular combination of parameter 
    # values in a 2d numpy.ndarray of shape (n_sample, n_coef), whose i, j element holds the 
    # i'th sample of the j'th coefficient. We store the arrays for different parameter values 
    # in a list of length n_test. 
    coef_samples : list[numpy.ndarray] = [sample_coefs(gp_list, candidate_parameters[i, :], trainer.n_samples) for i in range(n_candidates)];
    
    # Log GP prediction statistics for first candidate to diagnose zero-variance issue
    if n_candidates > 0:
        from GaussianProcess import eval_gp;
        pred_mean, pred_std = eval_gp(gp_list, candidate_parameters[0:1, :]);
        LOGGER.info("GP predictions for first candidate %s:" % str(candidate_parameters[0]));
        for coef_idx in range(min(5, pred_mean.shape[1])):
            LOGGER.info("  Coef %d: mean=%.6e, std=%.6e" % (coef_idx, pred_mean[0, coef_idx], pred_std[0, coef_idx]));
        avg_std = numpy.mean(pred_std[0, :]);
        LOGGER.info("  Average std across all coefficients: %.6e" % avg_std);
        if avg_std < 1e-6:
            LOGGER.warning("  WARNING: GP variance is near-zero! This suggests coefficients are nearly constant across parameter space!");
            LOGGER.warning("  This will cause poor greedy sampling - all points will look equally good/bad.");

    # Now, solve the latent dynamics forward in time for each set of coefficients in 
    # coef_samples. There are n_candidates combinations of parameter values, and we have 
    # n_samples sets of coefficients for each combination of parameter values. For the i'th one
    # of those, we want to solve the latent dynamics for n_t(i) times steps. Each solution 
    # frame consists of n_IC elements of \marthbb{R}^{n_z}.
    # 
    # Thus, we store the latent states in an n_candidates element list whose i'th element is an 
    # n_IC element list whose j'th element is an array of shape (n_samples, n_t(i), n_z) whose
    # p, q, r element holds the r'th component of j'th derivative of the latent state at the 
    # q'th time step when we use the p'th set of coefficient values sampled from the posterior
    # distribution for the i'th combination of testing parameter values.
    LatentStates    : list[list[numpy.ndarray]]     = [];
    n_z             : int                           = trainer.latent_dynamics.n_z;
    for i in range(n_candidates):
        LatentStates_i  : list[numpy.ndarray]    = [];
        for j in range(trainer.n_IC):
            # Each candidate can have a different number of time samples (adaptive stepping,
            # truncated outputs, etc.), so allocate per-candidate.
            LatentStates_i.append(numpy.empty([trainer.n_samples, len(t_Candidates[i]), n_z], dtype = numpy.float32));
        LatentStates.append(LatentStates_i);
    
    for i in range(n_candidates):
        # Fetch the t_Grid for the i'th combination of parameter values.
        # Use a 1D time grid (shared across ICs). This avoids accidental shape/length
        # mismatches due to 2D handling downstream.
        t_Grid  : numpy.ndarray = t_Candidates[i].detach().numpy().reshape(-1);

        # Simulate one sample at a time; store the resulting frames.           
        for j in range(trainer.n_samples):
            LatentState_ij : list[list[numpy.ndarray]] = trainer.latent_dynamics.simulate( 
                                                                coefs   = coef_samples[i][j:(j + 1), :], 
                                                                IC      = [Z0[i]], 
                                                                t_Grid  = [t_Grid], 
                                                                params  = candidate_parameters[i, :].reshape(1, -1));
            for k in range(trainer.n_IC):
                LatentStates[i][k][j, :, :] = LatentState_ij[0][k][:, 0, :];

    # Find the index of the parameter with the largest std.
    m_index : int = get_FOM_max_std(encoder_decoder, LatentStates);

    # We have found the testing parameter we want to add to the training set. Fetch it, then
    # stop the timer and return the parameter. 
    new_sample : numpy.ndarray = candidate_parameters[m_index, :].reshape(1, -1);
    LOGGER.info('New param: ' + str(numpy.round(new_sample, 4)) + '\n');
    trainer.timer.end("new_sample");

    # Now, append the new sample to the training set
    trainer.param_space.appendTrainSpace(new_sample);

    # Now that we know the new points we need to generate simulations for, we need to get ready to
    # actually run those simulations.
    return NextStep.RunSample;




def get_FOM_max_std(encoder_decoder : EncoderDecoder, LatentStates : list[list[numpy.ndarray]]) -> int:
    r"""
    We find the combination of parameter values which produces with FOM solution with the greatest
    variance.

    To make that more precise, consider the set of all FOM frames generated by decoding the latent 
    trajectories in LatentStates. We assume these latent trajectories were generated as follows:
    For a combination of parameter values, we sampled the posterior coefficient distribution for 
    that combination of parameter values. For each set of coefficients, we solved the corresponding
    latent dynamics forward in time. We assume the user used the same time grid for all latent 
    trajectories for that combination of parameter values.
    
    After solving, we end up with a collection of latent trajectories for that parameter value. 
    We then decoded each latent trajectory, which gives us a collection of FOM trajectories for 
    that combination of parameter values. At each value in the time grid, we have a collection of
    frames. We can compute the variance of each component of the frames at that time value for that
    combination of parameter values. We do this for each time value and for each combination of
    parameter values and then return the index for the combination of parameter values that gives
    the largest variance (among all components at all time frames).

    Stated another way, we find the following:
        argmax_{i}[ STD[ { Decoder(LatentStates[i][0][p, q, :])_k : p \in {1, 2, ... , n_samples} } ]
                    |   k \in {1, 2, ... , n_{FOM}},
                        i \in {1, 2, ... , n_param},
                        q \in {1, 2, ... , n_t(i)} ]
    

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    encoder_decoder : EncoderDecoder
        The encoder_decoder. We assume the solved dynamics (whose frames are stored in Zis) 
        take place in the encoder_decoder's latent space. We use this to decode the solution 
        frames.

    LatentStates : list[list[torch.Tensor]], len = n_param
        i'th element is an n_IC element list whose j'th element is a 3d tensor of shape 
        (n_samples, n_t(i), n_z) whose p, q, r element holds the r'th component of the j'th 
        component of the latent solution at the q'th time step when we solve the latent dynamics 
        using the p'th set of coefficients we got by sampling the posterior distribution for the 
        i'th combination of parameter values. 


    -----------------------------------------------------------------------------------------------
    Returns:
    -----------------------------------------------------------------------------------------------

    m_index : int
        The index of the testing parameter that gives the largest standard deviation. See the 
        description above for details.
    """
    
    # Run checks.
    assert isinstance(LatentStates,         list),                  "type(LatentStates) = %s, expected list" % (type(LatentStates));
    assert isinstance(LatentStates[0],      list),                  "type(LatentStates[0]) = %s, expected list" % (type(LatentStates[0]));
    assert isinstance(LatentStates[0][0],   numpy.ndarray),         "type(LatentStates[0][0]) = %s, expected numpy.ndarray" % (type(LatentStates[0][0]));
    assert len(LatentStates[0][0].shape)    == 3,                   "len(LatentStates[0][0].shape) = %d, expected 3" % (len(LatentStates[0][0].shape));

    n_param : int   = len(LatentStates);
    n_IC    : int   = len(LatentStates[0]);
    n_z     : int   = LatentStates[0][0].shape[2];

    assert n_z  == encoder_decoder.n_z, "n_z = %d, expected %d" % (n_z, encoder_decoder.n_z);

    for i in range(n_param):
        assert isinstance(LatentStates[i], list),                   "type(LatentStates[%d]) = %s, expected list" % (i, type(LatentStates[i]));
        assert len(LatentStates[i]) == n_IC,                        "len(LatentStates[%d]) = %d, expected %d" % (i, len(LatentStates[i]), n_IC);

        assert isinstance(LatentStates[i][0],   numpy.ndarray),     "type(LatentStates[%d][0]) = %s, expected numpy.ndarray" % (i, type(LatentStates[i][0]));
        assert len(LatentStates[i][0].shape)    == 3,               "len(LatentStates[%d][0].shape) = %d, expected 3" % (i, len(LatentStates[i][0].shape));
        n_samples_i : int   = LatentStates[i][0].shape[0];
        n_t_i       : int   = LatentStates[i][0].shape[1];

        for j in range(1, n_IC):
            assert isinstance(LatentStates[i][j],   numpy.ndarray), "type(LatentStates[%d][%d]) = %s, expected numpy.ndarray" % (i, j, type(LatentStates[i][j]));
            assert len(LatentStates[i][j].shape)    == 3,           "len(LatentStates[%d][%d].shape) = %d, expected 3" % (i, j, len(LatentStates[i][j].shape));
            assert LatentStates[i][j].shape[0]      == n_samples_i, "LatentStates[%d][%d].shape = %s, expected %d" % (i, j, str(LatentStates[i][j].shape), n_samples_i);
            assert LatentStates[i][j].shape[1]      == n_t_i,       "LatentStates[%d][%d].shape = %s, expected %d" % (i, j, str(LatentStates[i][j].shape), n_t_i);
            assert LatentStates[i][j].shape[2]      == n_z,         "LatentStates[%d][%d].shape = %s, expected %d" % (i, j, str(LatentStates[i][j].shape), n_z);


    # Find the index that gives the largest STD!
    max_std     : float     = 0.0;
    m_index     : int       = 0;
    
    if(isinstance(encoder_decoder, Autoencoder) or isinstance(encoder_decoder, CNN_3D_Autoencoder)):
        assert n_IC == 1, "n_IC = %d, expected 1" % (n_IC);

        for i in range(n_param):
            # Fetch the set of latent trajectories for the i'th combination of parameter values.
            # Z_i is a 3d tensor of shape (n_samples_i, n_t_i, n_z), where n_samples_i is the 
            # number of samples of the posterior distribution for the i'th combination of parameter 
            # values, n_t_i is the number of time steps in the latent dynamics solution for the 
            # i'th combination of parameter values, nd n_z is the dimension of the latent space. 
            # The p, q, r element of Zi is the r'th component of the q'th frame of the latent 
            # solution corresponding to p'th sample of the posterior distribution for the i'th 
            # combination of parameter values.
            Z_i             : torch.Tensor  = torch.Tensor(LatentStates[i][0]);

            # Now decode the frames, one sample at a time.
            n_samples_i     : int           = Z_i.shape[0];
            n_t_i           : int           = Z_i.shape[1];
            if isinstance(encoder_decoder, CNN_3D_Autoencoder):
                fom_shape = [encoder_decoder.conv_channels[0]] + encoder_decoder.reshape_shape   # [C,I,J,K]
            else:
                fom_shape = encoder_decoder.reshape_shape

            U_Pred_i = numpy.empty([n_samples_i, n_t_i] + fom_shape, dtype = numpy.float32)
            for j in range(n_samples_i):
                U_Pred_i[j, ...] = encoder_decoder.Decode(Z_i[j, :, :])[0].detach().numpy();

            # Compute the standard deviation across the sample axis. This gives us an array of shape 
            # (n_t, n_FOM) whose i,j element holds the (sample) standard deviation of the j'th component 
            # of the i'th frame of the FOM solution. In this case, the sample distribution consists of 
            # the set of j'th components of i'th frames of FOM solutions (one for each sample of the 
            # coefficient posterior distributions).
            U_pred_i_std    : numpy.ndarray = U_Pred_i.std(0);

            # Handle inf/nan values gracefully by replacing them with a large but finite value
            if not numpy.all(numpy.isfinite(U_pred_i_std)):
                LOGGER.warning(f"Parameter {i}: STD contains inf/nan values. This suggests divergent samples escaped detection. Replacing with 1e10.");
                U_pred_i_std = numpy.nan_to_num(U_pred_i_std, nan = 0.0, posinf = 1e10, neginf = 1e10);
            
            # Compute the maximum standard deviation 
            max_std_i                : numpy.float32 = U_pred_i_std.max();

            # If this is bigger than the biggest std we have seen so far, update the maximum.
            if max_std_i > max_std:
                m_index : int   = i;
                max_std : float = max_std_i;

        # Report the index of the testing parameter that gave the largest maximum std.
        return m_index


    elif(isinstance(encoder_decoder, Autoencoder_Pair)):
        assert n_IC == 2, "n_IC = %d, expected 2" % (n_IC);

        for i in range(n_param):
            # Fetch the set of latent trajectories for the i'th combination of parameter values.
            # Z_D_i and Z_D_i are a 3d tensor sof shape (n_samples_i, n_t_i, n_z), where 
            # n_samples_i is the number of samples of the posterior distribution for the i'th 
            # combination of parameter values, n_t_i is the number of time steps in the latent 
            # dynamics solution for the i'th combination of parameter values, nd n_z is the 
            # dimension of the latent space. 
            # 
            # The p, q, r element of Z_D_i is the r'th component of the q'th frame of the latent 
            # displacement corresponding to p'th sample of the posterior distribution for the i'th 
            # combination of parameter values. The components of Z_V_i are analogous but for the 
            # latent velocity. 
            Z_D_i   : torch.Tensor  = torch.Tensor(LatentStates[i][0]);
            Z_V_i   : torch.Tensor  = torch.Tensor(LatentStates[i][1]);

            n_samples_i : int           = Z_D_i.shape[0];
            n_t_i       : int           = Z_D_i.shape[1];
            D_Pred_i    : numpy.ndarray = numpy.empty([n_samples_i, n_t_i] + encoder_decoder.reshape_shape, dtype = numpy.float32);
            for j in range(n_samples_i):
                D_Pred_ij, _ = encoder_decoder.Decode(Latent_Displacement   = Z_D_i[j, :, :], Latent_Velocity    = Z_V_i[j, :, :]);
                D_Pred_i[j, ...] = D_Pred_ij.detach().numpy();

            # Compute the standard deviation across the sample axis. This gives us an array of 
            # shape (n_t, n_FOM) whose i,j element holds the (sample) standard deviation of the 
            # j'th component of the i'th frame of the FOM solution. In this case, the sample 
            # distribution consists of the set of j'th components of i'th frames of FOM solutions 
            # (one for each sample of the coefficient posterior distributions).
            D_Pred_i_std    : numpy.ndarray = D_Pred_i.std(0);

            # Handle inf/nan values gracefully by replacing them with a large but finite value
            if not numpy.all(numpy.isfinite(D_Pred_i_std)):
                LOGGER.warning(f"Parameter {i}: STD contains inf/nan values. This suggests divergent samples escaped detection. Replacing with 1e10.");
                D_Pred_i_std = numpy.nan_to_num(D_Pred_i_std, nan=0.0, posinf=1e10, neginf=1e10);
            
            # Compute the maximum standard deviation.
            max_std_i                : numpy.float32 = D_Pred_i_std.max();

            # If this is bigger than the biggest std we have seen so far, update the maximum.
            if max_std_i > max_std:
                m_index : int   = i;
                max_std : float = max_std_i;

        # Report the index of the testing parameter that gave the largest maximum std.
        return m_index;
    
    
    else:
        raise ValueError("Invalid EncoderDecoder type!");
