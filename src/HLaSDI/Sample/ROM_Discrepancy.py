# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  logging;
import  time;

import  torch;
import  numpy;

from    HLaSDI.Enums                       import  NextStep;  
from    HLaSDI.Trainer                     import  Trainer;
from    HLaSDI.Rollouts                    import  Mean_Rollout;
from    HLaSDI.EncoderDecoder              import  EncoderDecoder;
from    HLaSDI.Schemas                     import  ROMDiscrepancySamplerConfig;
from    HLaSDI.Sample.Sampler              import  Sampler;


# Setup logger.
LOGGER : logging.Logger = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# ROM_Discrepancy class
# -------------------------------------------------------------------------------------------------

class ROM_Discrepancy(Sampler):
    def __init__(self, config : ROMDiscrepancySamplerConfig):
        r"""
        Initializes a "ROM_Discrepancy" Sampler object. This class defines the "worst" parameter 
        as the testing parameter combination (outside of the training set) whose interpolated mean 
        latent dynamics is most different from every latent dynamics already in the training set.

        For a candidate testing parameter \theta_i and a training parameter \theta_j, this sampler 
        first solves the mean latent dynamics for \theta_i on the time grid used by the \theta_j 
        training solution. It then evaluates both right hand sides along that candidate trajectory:
        the interpolated mean RHS for \theta_i and the fixed/training RHS for \theta_j. Their 
        discrepancy is the time average of the Euclidean norm of this RHS difference, computed with
        the trapezoidal rule. The sampler assigns each candidate the minimum discrepancy across all 
        training parameters and appends the candidate with the largest such minimum discrepancy.

        ROM_Discrepancy is non-intrusive with respect to held-out FOM trajectories: it uses the 
        testing parameter grid and the learned/interpolated ROM, but it does not compare decoded 
        predictions against true testing solutions.

        

        -------------------------------------------------------------------------------------------
        Arguments:
        -------------------------------------------------------------------------------------------

        config: ROMDiscrepancySamplerConfig
            The 'sampler' portion of the .yml configuration file. Should contain a 'type' 
            attribute whose value is "ROM_Discrepancy".
        """
        
        # Schema validation happens at configuration load time, so this check is only a boundary
        # assertion that Initialize passed the right sampler schema object.
        assert isinstance(config, ROMDiscrepancySamplerConfig), "config object SamplerConfig, got %s" % str(type(config))

        super().__init__(requires_stochastic_LD = False, config = config);


    def Sample(self, trainer : Trainer) -> NextStep:
        r"""
        This function identifies the combination of testing parameters (which are not in the 
        training set) that has the maximum minimum discrepancy with a training parameter. 

        Specifically, for each parameter, \theta, let T_{\theta} denote the length of the time 
        interval for that parameter. For each testing parameter, \theta_i, and training parameter, 
        \theta_j, we solve the (mean) latent dynamics for \theta_i on the time grid used by the 
        \theta_j training solution. Let Z_{i,j} denote this solution. We then define the discrepancy
        between the i'th candidate and the j'th training parameter, denoted D_{i,j}, as the mean 
        difference between the RHS of the fixed/training dynamics and the RHS of the interpolated 
        mean testing dynamics when evaluated along the mean testing solution. That is,
         
           D_{i,j} = (1/T_j) \int_{0}^{T_j} || \mean{f_{\theta_i}}(Z_{i,j}(t), t, \theta_i) - f_{\theta_j}(Z_{i,j}(t), t, \theta_j) ||

        Here, T_j is the length of the time interval on which we have measurements of the j'th 
        training solution, f_{\theta_j} denotes the learned latent dynamics coefficients for the 
        j'th training parameter, and \mean{f_{\theta_i}} denotes the interpolated mean latent 
        dynamics for the i'th testing parameter.

        In practice, we solve the mean LD for the i'th candidate at the times at which we have the 
        j'th training parameter. We also evaluate this integral using the trapezoidal rule. 

        We then select the candidate given by satisfies 

            argmax_i min_j D_{i, j}


        
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

        new_sample_timer : float = time.perf_counter();
        n_test          : int               = trainer.param_space.n_test();
        n_train         : int               = trainer.param_space.n_train();
        assert n_test > 0, "trainer.param_space.n_test() = %d" % n_test;
        LOGGER.info('\n~~~~~~~ Finding New Point ~~~~~~~');

        # Move the encoder_decoder to the cpu (this is where all the GP stuff happens). Remember 
        # that train_coefs should specify the coefficients from that iteration. 
        encoder_decoder_device : torch.device = next(trainer.encoder_decoder.parameters()).device;
        encoder_decoder : EncoderDecoder    = trainer.encoder_decoder.cpu();


        # ---------------------------------------------------------------------------------------------
        # Find the candidate parameters ({test set} - {train set}).

        # Find the candidate parameters (the elements of the testing set not in the training set).
        candidate_parameters    : list[numpy.ndarray]       = [];
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
        
        # Concatenate the candidates to form an array of shape (n_candidates, n_param).
        n_candidates : int = len(candidate_parameters);
        LOGGER.info("There are %d candidate testing parameters (%d in the testing space, %d in the training set)" % (n_candidates, n_test, n_train));
        assert n_candidates >= 1, "n_candidates = %d" % n_candidates;
        candidate_parameters    = numpy.array(candidate_parameters);


        # -----------------------------------------------------------------------------------------
        # Compute ROM discrepancy for each testing parameter. 

        LOGGER.debug("Setting up arrays to hold ROM discrepancies");
        n_param     : int   = candidate_parameters.shape[1];

        # Build one-dimensional numpy time grids for RHS evaluation and trapezoidal integration. 
        # Mean_Rollout accepts torch tensors or numpy arrays, but RHS methods expect 1D time grids and
        # the quadrature below should use numpy arrays regardless of how the trainer stores times.
        t_Train_np : list[numpy.ndarray] = [];
        for j in range(n_train):
            if(isinstance(trainer.t_Train[j], torch.Tensor)):
                t_Train_np.append(trainer.t_Train[j].detach().cpu().numpy().reshape(-1));
            else:
                t_Train_np.append(numpy.asarray(trainer.t_Train[j]).reshape(-1));

        # i'th element specifies the minimum over j of the sum (across samples) ROM discrepancy 
        # between the RHS of the mean latent dynamics for the i'th candidate and the 
        # latent dynamics of the j'th training parameter when evaluated on the the latent 
        # trajectory we get by solving mean latent dynamics for the i'th candidate along
        # the times for the j'th training parameter. That is,
        #
        #   Discrepancies[i] = min_{j} (1/T_j) \int_{0}^{T_j} || mean(f(Z_{i,j}(t), t, \theta_i)) - f(Z_{i,j}(t), t, \theta_j) ||_2 dt
        #
        # Where \theta_i = i'th candidate parameter, \theta_j is the j'th training parameter, 
        # T_j is the time interval length for the jth training parameter, and Z_{i,j}(t) denotes
        # the solution to the mean latent dynamics for the i'th candidate on the j'th training time
        # grid.
        Discrepancies : numpy.ndarray = numpy.zeros((n_candidates), dtype = numpy.float32);
        rollout_time : float = 0.0;
        scoring_time : float = 0.0;
        for i in range(n_candidates):
            # broadcast the i'th parameter to have n_train copies of itself; this way, we can use
            # Mean_Rollout to solve the LD along each training time grid. 
            broadcast_ith_candidate = numpy.broadcast_to(candidate_parameters[i, :], (n_train, n_param));

            # Solve the LD for the i'th candidate at each time training time grid. 
            rollout_timer : float = time.perf_counter();
            Zis_Mean : list[list[numpy.ndarray]] = Mean_Rollout(
                                                    encoder_decoder     = encoder_decoder, 
                                                    physics             = trainer.physics,
                                                    latent_dynamics     = trainer.latent_dynamics, 
                                                    param_grid          = broadcast_ith_candidate, 
                                                    t_Grid              = t_Train_np, 
                                                    trainer             = trainer);  
            rollout_time += time.perf_counter() - rollout_timer;

            # Evaluate the RHS of the latent dynamics for each training parameter along these
            # trajectories.
            scoring_timer : float = time.perf_counter();
            RHS_training : list[numpy.ndarray] = trainer.latent_dynamics.RHS(
                                                    Z       = Zis_Mean,
                                                    t_Grid  = t_Train_np,
                                                    params  = trainer.param_space.train_space,
                                                    sample  = False);

            # Likewise, evaluate the RHS of the mean LD for the current candidate.
            RHS_Candidate : list[numpy.ndarray] = trainer.latent_dynamics.RHS(
                                                    Z       = Zis_Mean,
                                                    t_Grid  = t_Train_np,
                                                    params  = broadcast_ith_candidate,
                                                    sample  = False);

            # Compute the discrepancy (using trapezoidal rule)) for each training parameter.
            ith_Discrepancies : numpy.ndarray = numpy.empty((n_train), dtype = numpy.float32);
            for j in range(n_train):
                # We want to compute the following:
                #
                #   ith_Discrepancies[j] = (1/T_j) \sum_{k = 1}^{n_t_j} (t_k - t_{k - 1})*(1/2)*(R_{i,j}(t_k) + R_{i,j}(t_{k - 1}))
                # 
                # Where, R_{i,j}(t) = || mean(f(Z_{i,j}(t), t, \theta_i)) - f(Z_{i,j}(t), t, \theta_j) ||_2

                # Fetch the times for the j'th training parameter.
                jth_t_Grid  : numpy.ndarray = t_Train_np[j];
                T_j         : float         = float(jth_t_Grid[-1] - jth_t_Grid[0]);
                assert T_j > 0., "Training time grid %d must span a positive time interval, got %g" % (j, T_j);

                # Compute || mean(f(Z_{i,j}(t), t, \theta_i)) - f(Z_{i,j}(t), t, \theta_j) ||_2 at each time.
                ij_integrand : numpy.ndarray = numpy.linalg.norm(RHS_training[j] - RHS_Candidate[j], ord = 2, axis = -1);

                # Compute step sizes.
                jth_step_sizes = jth_t_Grid[1:] - jth_t_Grid[:-1];

                # Compute average of integrand on left and right hand side of each step.
                jth_integrand_step_averages : numpy.ndarray = (ij_integrand[1:] + ij_integrand[:-1])/2;

                # Use trapezoidal rule!
                ith_Discrepancies[j] = (1./T_j)*numpy.dot(jth_step_sizes, jth_integrand_step_averages);

            # The discrepancy for the i'th candidate is the minimum across training parameters.
            Discrepancies[i] = numpy.min(ith_Discrepancies);
            scoring_time += time.perf_counter() - scoring_timer;


        # ---------------------------------------------------------------------------------------------
        # Wrap up.

        # Find the candidate with the maximum discrepancy.
        index : int = int(numpy.argmax(Discrepancies));

        # Move the model back to its original device now that ROM discrepancy computation is done.
        trainer.encoder_decoder.to(encoder_decoder_device);

        # We have found the testing parameter we want to add to the training set. Fetch it, then
        # stop the timer and return the parameter. 
        new_sample : numpy.ndarray = candidate_parameters[index, :].reshape(1, -1);
        LOGGER.info('New param: ' + str(numpy.round(new_sample, 4)) + '\n');
        trainer._cache_metric("time/new_sample",        time.perf_counter() - new_sample_timer);
        trainer._cache_metric("sampler/n_candidates",   n_candidates);
        trainer._cache_metric("sampler/score/selected", Discrepancies[index]);
        trainer._cache_metric("sampler/score/max",      numpy.max(Discrepancies));
        trainer._cache_metric("sampler/score/mean",     numpy.mean(Discrepancies));
        trainer._cache_metric("sampler/score/std",      numpy.std(Discrepancies));
        trainer._cache_metric("sampler/score/min",      numpy.min(Discrepancies));
        trainer._cache_metric("time/sampler/rollout",   rollout_time);
        trainer._cache_metric("time/sampler/scoring",   scoring_time);
        trainer._flush_metrics_cache(trainer.restart_iter);

        # Now, append the new sample to the training set
        trainer.param_space.appendTrainSpace(new_sample);

        # Now that we know the new points we need to generate simulations for, we need to get ready to
        # actually run those simulations.
        return NextStep.RunSample;
