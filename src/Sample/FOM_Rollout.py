# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

# Add the main (src) directory to the search path.
import  os, sys;
src_path            : str   = os.path.abspath(os.path.dirname(os.path.dirname(__file__)));
EncoderDecoder_Path : str   = os.path.join(src_path, "EncoderDecoder");
Utilities_path      : str   = os.path.join(src_path, "Utilities");
sys.path.append(Utilities_path);
sys.path.append(src_path);

import  logging;

from    sklearn.gaussian_process    import  GaussianProcessRegressor;
import  torch;
import  numpy;

from    Enums                       import  NextStep;  
from    Trainer                     import  Trainer;
from    SolveROMs                   import  sample_roms;
from    GaussianProcess             import  fit_gps;
from    EncoderDecoder              import  EncoderDecoder;


# Setup logger.
LOGGER : logging.Logger = logging.getLogger(__name__);




# -------------------------------------------------------------------------------------------------
# FOM_Rollout function
# -------------------------------------------------------------------------------------------------

def FOM_Rollout(trainer : Trainer) -> NextStep:
    """
    This function identifies the combination of testing parameters (which are not in the training 
    set) that has the average (across n_samples samples) rollout error with the corresponding 
    true solution. We add this combination of parameters to the training set, then hand it off 
    ready to generate the new training solution (and add it to the trainer).

    How this works is fairly simple: We begin by finding every testing parameter which is not 
    in the training set. For each parameter combination, we encode its IC and draw n_sample 
    samples of the posterior distribution for the coefficients evaluated at the current combination
    of parameters. This gives us n_samples IVPs in the latent space, which we solve forward in 
    time to predict the future latent states. We decode these trajectories and compare each one 
    to the ground truth for this combination of parameter values. 
    
    Next, we compute the sum (across samples, time derivatives/number of ICs, and time steps) of 
    the relative errors between the samples and the corresponding ground truth. We do this for 
    each parameter combination, then select the one which gives the largest value. This is the 
    parameter combination that we add to the training set. 


    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    Trainer : Trainer
        The trainer object we use throughout this process.



    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    A NextStep.RunSample object indicating we are ready to add the solution for the new training 
    parameter to the trainer's U_Train attribute.
    """

    # ---------------------------------------------------------------------------------------------
    # Setup

    trainer.timer.start("new_sample");
    assert len(trainer.U_Test)             >  0,                                    "len(trainer.U_Test) = %d" % len(trainer.U_Test);
    assert len(trainer.U_Test)             == trainer.param_space.n_test(),         "len(trainer.U_Test) = %d, trainer.param_space.n_test() = %d" % (len(trainer.U_Test), trainer.param_space.n_test());
    assert trainer.best_train_coefs is not None,                                    "best_train_coefs is None (did training run and checkpoint succeed?)";
    assert trainer.best_train_coefs.shape[0]     == trainer.param_space.n_train(),  "trainer.best_train_coefs.shape[0] = %d, trainer.param_space.n_train() = %d" % (trainer.best_train_coefs.shape[0], trainer.param_space.n_train());

    train_coefs : numpy.ndarray = trainer.best_train_coefs;                     # Shape = (n_train, n_coefs).
    LOGGER.info('\n~~~~~~~ Finding New Point ~~~~~~~');

    # Move the encoder_decoder to the cpu (this is where all the GP stuff happens) and load the 
    # encoder_decoder from the last checkpoint. This should be the one that obtained the best 
    # loss so far. Remember that train_coefs should specify the coefficients from that iteration. 
    LOGGER.info("Sampling: Loadin encoder_decoder from checkpoint.");
    encoder_decoder : EncoderDecoder    = trainer.encoder_decoder.cpu();
    n_test          : int               = trainer.param_space.n_test();
    n_train         : int               = trainer.param_space.n_train();
    encoder_decoder.load_state_dict(torch.load(trainer.path_checkpoint + '/' + 'checkpoint.pt', map_location = 'cpu'));



    # ---------------------------------------------------------------------------------------------
    # Find the candidiate parameters ({test set} - {train set}).

    # Find the candidate parameters (the elements of the testing set not in the training set).
    candidate_parameters    : list[numpy.ndarray]       = [];
    t_Candidates            : list[torch.Tensor]        = [];
    U_Candidates            : list[list[torch.Tensor]]  = [];
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
            U_Candidates.append(trainer.U_Test[i]);
    
    # Concatenate the candidates to form an array of shape (n_candidates, n_coefs).
    n_candidates : int = len(candidate_parameters);
    LOGGER.info("There are %d candidate testing parameters (%d in the testing space, %d in the training set)" % (n_candidates, n_test, n_train));
    assert n_candidates >= 1, "n_candidates = %d" % n_candidates;
    candidate_parameters    = numpy.array(candidate_parameters);



    # ---------------------------------------------------------------------------------------------
    # Fit the GPs

    # Log coefficient statistics before fitting GPs
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



    # ---------------------------------------------------------------------------------------------
    # Generate the latent trajectories.

    LOGGER.debug("Sampling roms with %d rollouts per candidiate" % trainer.n_samples);
    Zis_Samples : list[list[torch.Tensor]] = sample_roms(
                                                encoder_decoder     = encoder_decoder, 
                                                physics             = trainer.physics,
                                                latent_dynamics     = trainer.latent_dynamics, 
                                                gp_list             = gp_list, 
                                                param_grid          = candidate_parameters, 
                                                t_Grid              = t_Candidates, 
                                                n_samples           = trainer.n_samples, 
                                                trainer             = trainer);
    

    # ---------------------------------------------------------------------------------------------
    # Decode the samples and compute relative errors

    LOGGER.debug("Setting up arrays to hold relative errors");
    n_samples   : int   = trainer.n_samples;
    n_IC        : int   = trainer.n_IC;

    # i'th element is an n_IC element list whose j'th element is an array of shape
    # n_samples x n_t_i shape array whose [p, q] element holds the relative error between 
    # the p'th sample's prediction of the j'th time derivative of the FOM solution corresponding 
    # to the i'th combination of candidiate parameters at the q'th time step.
    Rel_Error   : list[list[numpy.ndarray]] = [];                   # (n_candidates)

    for i in range(n_candidates):
        # Initialize lists for the i'th combination of parameter values
        Rel_Error_i : list[numpy.ndarray]   = [];                   # (n_IC)

        # Fetch n_t_i.
        n_t_i : int = t_Candidates[i].shape[0];

        # Build an array for each derivative of the FOM solution.
        for j in range(n_IC):
            Rel_Error_i.append(numpy.zeros((n_samples, n_t_i), dtype = numpy.float32));

        # Append the lists for the i'th combination to the overall lists.
        Rel_Error.append(Rel_Error_i);
    


    # ---------------------------------------------------------------------------------------------
    # Compute relative errors.

    # If the workflow uses normalization, U_Test and decoded predictions are in normalized
    # units. De-normalize here for meaningful physical errors/plots using the trainer.
    use_denorm : bool = hasattr(trainer, "has_normalization") and trainer.has_normalization();

    # Find the index that gives the largest total relative error
    max_Total_Rel_Error : float = 0.0;
    m_index             : int   = 0;
    eps                 : float = 0.01;

    for i in range(n_candidates):
        # Setup
        n_t_i       : int                   = t_Candidates[i].shape[0];
        U_Cand_i    : list[torch.Tensor]    = U_Candidates[i];              # (n_IC)

        # Decode the samples for the j'th candidiate, compute relative error.
        for j in range(n_samples):
            # Decode the j'th set of samples for the i'th combination of parameters.
            Zis_sample_ij: list[torch.Tensor] = [];
            for k in range(n_IC):
                Zis_sample_ij.append(torch.Tensor(Zis_Samples[i][k][:, j, :]));
            U_Pred_ij           : tuple[torch.Tensor]   = encoder_decoder.Decode(*Zis_sample_ij);
            
            # Set up a list to hold the STDs of the FOM solution.
            U_Cand_i_std        : list[float]           = [];

            # Convert to numpy and denormalize. Also populate U_Test_i_std.
            U_Pred_ij_np        : list[numpy.ndarray]   = [];
            U_Cand_i_np         : list[numpy.ndarray]   = [];
            for k in range(n_IC):
                U_Pred_ij_np.append(U_Pred_ij[k].detach().numpy());
                U_Cand_i_np.append( U_Cand_i[k].detach().numpy());          # (n_t_i, physics.Frame_Shape)
                
                if use_denorm:
                    U_Pred_ij_np[k]     = trainer.denormalize_np(U_Pred_ij_np[k], k);
                    U_Cand_i_np[k]      = trainer.denormalize_np(U_Cand_i_np[k], k);
            
                U_Cand_i_std.append(numpy.std(U_Cand_i_np[k]));
                if(U_Cand_i_std[k] < eps):
                    LOGGER.warning("The std for the %d'th candidiate (%s) is below %f; replacing with %f" % (k, str(candidate_parameters[k]), eps, eps));
                    U_Cand_i_std[k] = eps;

            # For each frame, compute the relative error between the true and predicted FOM solutions.
            # We normalize the error by the std of the true solution.
            for p in range(n_IC):
                for q in range(n_t_i):
                    Rel_Error[i][p][j, q] = numpy.mean(numpy.abs(U_Pred_ij_np[p][q, ...] - U_Cand_i_np[p][q, ...]))/U_Cand_i_std[p];
    
        # Now, Total_Rel_Error[i] should hold the sum of relative errors across time derivatives, 
        # time steps, and samples.
        Total_Rel_Error_i = 0.0;
        for p in range(n_IC):
            Total_Rel_Error_i += numpy.sum(Rel_Error[i][p]);
    

        # If this is bigger than the biggest total relative error we have seen so far, update the 
        # maximum and corresponding index.
        if(Total_Rel_Error_i > max_Total_Rel_Error):
            LOGGER.info("Found new largest total relative error (%f) with parameter combination %s" % (Total_Rel_Error_i, str(candidate_parameters[k])));
            max_Total_Rel_Error = Total_Rel_Error_i;
            m_index             = i;



    # ---------------------------------------------------------------------------------------------
    # Wrap up.

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