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
from    Sampler                     import  Sampler;


# Setup logger.
LOGGER : logging.Logger = logging.getLogger(__name__);




# -------------------------------------------------------------------------------------------------
# FOM_Rollout class
# -------------------------------------------------------------------------------------------------

class FOM_Rollout(Sampler):
    def __init__(self, config : dict):
        """
        Initializes a "FOM_Rollout" Sampler object. This class defines the "worst" parameter 
        as the testing parameter combination (outside of the training set) that produces the 
        largest average (computed empirically using samples of the coefficient posterior 
        distribution) rollout error. This works by first computing the absolute rollout error,
        then optionally normalizing it to get a relative error.

        This is an intrusive sampler in that it assumes we have access to the true solution for 
        every parameter combination in the testing set.
        
        A FOM_Rollout object has a few settings: `n_samples`, `normalized_FOM`, and 
        `error_normalization`.

        `n_samples` is an integer specifying the number of samples we draw from the coefficient 
        posterior distribution (see above). 

        `normalized_FOM` is a boolean specifying if we should use normalized or denormalized 
        FOM values to compute the absolute error.

        `error_normalization` specifies how we should normalize the absolute error. This can 
        be `none` (use absolute error; best when using normalized absolute error), `global_std` 
        (uses the standard deviation of U_Train for the p'th IC as the normalizing factor for 
        the p'th component of the absolute error) or `trajectory_std` (which divides the absolute
        error for the p'th IC of the current parameter combination by the std of the p'th component
        of the true solution).

        

        -------------------------------------------------------------------------------------------
        Arguments:
        -------------------------------------------------------------------------------------------

        config: dict
            The 'sampler' portion of the .yml configuration file. Should contain a 'type' 
            attribute whose value is "FOM_Rollout", as well as a "FOM_Rollout" key whose value 
            is a dictionary with three keys: `normalized_FOM`, and `error_normalization`. 
            See above.
        """
        # Checks
        super().__init__(config);

        assert 'FOM_Rollout' in config, "sampler config must contain a 'FOM_Rollout' key";
        sub = config['FOM_Rollout'];
        assert isinstance(sub, dict), "sampler.FOM_Rollout must be a dict";
        self.n_samples : int = int(sub['n_samples']);

        # Config key: normalized_FOM (bool). If True, compute errors in normalized units;
        # if False, compute errors in physical units (requires trainer normalization stats).
        assert 'normalized_FOM' in sub, "FOM_Rollout config must include boolean key normalized_FOM";
        self.normalized_FOM : bool = bool(sub['normalized_FOM']);

        # Config key: error_normalization
        assert 'error_normalization' in sub, "FOM_Rollout config must include key error_normalization";
        self.error_normalization : str = str(sub['error_normalization']).lower();
        assert self.error_normalization in ['none', 'global_std', 'trajectory_std'], (
            f"FOM_Rollout.error_normalization must be none|global_std|trajectory_std, got {self.error_normalization}"
        );

        # Optional epsilon used for std-based normalizations
        self.eps : float = float(sub.get('eps', 0.01));


    def Sample(self, trainer : Trainer) -> NextStep:
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
        # Find the candidate parameters ({test set} - {train set}).

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

        LOGGER.debug("Sampling roms with %d rollouts per candidate" % self.n_samples);
        Zis_Samples : list[list[torch.Tensor]] = sample_roms(
                                                    encoder_decoder     = encoder_decoder, 
                                                    physics             = trainer.physics,
                                                    latent_dynamics     = trainer.latent_dynamics, 
                                                    gp_list             = gp_list, 
                                                    param_grid          = candidate_parameters, 
                                                    t_Grid              = t_Candidates, 
                                                    n_samples           = self.n_samples, 
                                                    trainer             = trainer);
        

        # ---------------------------------------------------------------------------------------------
        # Decode the samples and compute relative errors

        LOGGER.debug("Setting up arrays to hold relative errors");
        n_samples   : int   = self.n_samples;
        n_IC        : int   = trainer.n_IC;

        # i'th element is an n_IC element list whose j'th element is an array of shape
        # n_samples x n_t_i shape array whose [p, q] element holds the relative error between 
        # the p'th sample's prediction of the j'th time derivative of the FOM solution corresponding 
        # to the i'th combination of candidate parameters at the q'th time step.
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
        # Precompute global denominators per IC if requested.

        global_denoms = None
        if self.error_normalization == 'global_std':
            if self.normalized_FOM == True:
                # Use the current (possibly normalized) stored test set directly.
                _, stds = trainer._compute_mean_std_from_U(trainer.U_Train);
                global_denoms = [max(float(s), self.eps) for s in stds];
            else:  # denormalized
                if hasattr(trainer, 'has_normalization') and trainer.has_normalization():
                    # If normalization is affine: std_phys = std_norm * data_std. For test-normalized data, std_norm ~ 1.
                    global_denoms = [max(float(trainer.data_std[j].item()), self.eps) for j in range(trainer.n_IC)];
                else:
                    _, stds = trainer._compute_mean_std_from_U(trainer.U_Train);
                    global_denoms = [max(float(s), self.eps) for s in stds];


        # ---------------------------------------------------------------------------------------------
        # Compute relative errors.

        max_Total_Rel_Error : float = -1.0;
        m_index             : int   = 0;

        # Candidate_parameters is a list[ndarray]; make it an ndarray for indexing/logging.
        candidate_parameters = numpy.asarray(candidate_parameters);

        for i in range(n_candidates):
            n_t_i       : int                   = t_Candidates[i].shape[0]
            U_Cand_i    : list[torch.Tensor]    = U_Candidates[i]  # (n_IC)

            # Prepare true trajectory arrays in requested units.
            U_true_np : list[numpy.ndarray] = []
            for p in range(n_IC):
                arr = U_Cand_i[p].detach().numpy()
                if (self.normalized_FOM == False) and hasattr(trainer, 'has_normalization') and trainer.has_normalization():
                    arr = trainer.denormalize_np(arr, p)
                U_true_np.append(arr)

            # Compute trajectory-specific denominators if requested.
            traj_denoms = None;
            if self.error_normalization == 'trajectory_std':
                traj_denoms = [max(float(numpy.std(U_true_np[p])), self.eps) for p in range(n_IC)]

            # Accumulate a score: avg over samples at each time -> max over time -> sum over ICs.
            Total_i = 0.0
            for p in range(n_IC):
                # Compute denominator
                if self.error_normalization == 'none':
                    denom_p = 1.0;
                elif self.error_normalization == 'global_std':
                    denom_p = float(global_denoms[p]);
                else:  # trajectory_std
                    denom_p = float(traj_denoms[p]);

                # Build per-sample per-time errors for this derivative.
                err_samples = numpy.zeros((n_samples, n_t_i), dtype = numpy.float32)
                for j in range(n_samples):
                    # Decode the j'th set of samples for the i'th combination of parameters.
                    Zis_sample_ij : list[torch.Tensor] = [];
                    for k in range(n_IC):
                        Zis_sample_ij.append(torch.Tensor(Zis_Samples[i][k][:, j, :]));
                    U_Pred_ij : tuple[torch.Tensor] = encoder_decoder.Decode(*Zis_sample_ij);

                    # Convert to numpy and denormalize.
                    U_pred_np = U_Pred_ij[p].detach().numpy();
                    if self.normalized_FOM == False and hasattr(trainer, 'has_normalization') and trainer.has_normalization():
                        U_pred_np = trainer.denormalize_np(U_pred_np, p);

                    # For each frame, compute the mean absolute error across the spatial dimensions 
                    # (vectorized over spatial dims; loop only over time) between the true and 
                    # predicted FOM solutions, then divide by denom_p (which may be candidate 
                    # specific).
                    for t_idx in range(n_t_i):
                        err_samples[j, t_idx] = numpy.mean(numpy.abs(U_pred_np[t_idx, ...] - U_true_np[p][t_idx, ...])) / denom_p;

                # Average across the samples.
                mean_over_samples = numpy.mean(err_samples, axis = 0);

                # Add the contribution for the p'th IC to the total for the i'th candidate.
                Total_i += float(numpy.max(mean_over_samples));

            # Check if we have a new "worst" parameter combination.
            if Total_i > max_Total_Rel_Error:
                LOGGER.info("Found new largest sampling score (%f) with parameter combination %s" % (Total_i, str(candidate_parameters[i])));
                max_Total_Rel_Error = Total_i;
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