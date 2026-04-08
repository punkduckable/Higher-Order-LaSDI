# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

# Add the main (src) directory to the search path.
import  os, sys;
src_path        : str   = os.path.abspath(os.path.dirname(os.path.dirname(__file__)));
sys.path.append(src_path);

import  numpy;
import  torch;

from    Trainer                     import  Trainer;
from    Enums                       import  NextStep;  

import  logging;

# Setup logger.
LOGGER : logging.Logger = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# Sampler class
# -------------------------------------------------------------------------------------------------

class Sampler:
    def __init__(self, config : dict):
        """
        The Sampler class defines how Greedy Sampling picks new parameter combinations. 
        We use it at the end of each round of training to pick the "worst" parameter combination, 
        which we then add to the training set. 

        Samplers can be "intrusive" or "non-intrusive". Intrusive samplers require the true 
        solution for each testing parameter combination while non-intrusive ones do not.

        Defining a Sampler sub-class is fairly simple: All you need to do is define the initializer
        and the "Sample" method. The initializer should simply pass the Sampler config to the 
        Sampler super class, while the "Sample" method should take a trainer object and use it to 
        find the "worst" parameter combination (which it should then return). 

        The Sampler base class also includes a "Generate" method which actually generates the 
        training solution for the new (worst) parameter combination and appends this new solution 
        onto the trainer's U_Train and t_Train attributes.



        -------------------------------------------------------------------------------------------
        Arguments:
        -------------------------------------------------------------------------------------------

        config: dict
            The 'sampler' portion of the .yml configuration file. Should contain a 'type' 
            attribute.
        """

        # Check that there is "type" attribute in the config.
        assert isinstance(config, dict),            "config must be a dictionary! got %s" % str(type(config));
        assert 'type'           in config.keys(),   "sampler config must have a `type` key!";
        assert isinstance(config['type'], str),     "sampler type must be a str, not %s" % str(type(config['type']));

        # store the config.
        self.config : dict  = config;
        self.type   : str   = config['type']; 


    def Sample(self, trainer : Trainer) -> NextStep:
        """
        This function should take in a trainer object and identifies the "worst" combination of 
        testing parameters (which are not in the training set). The definition of "worst" is up to
        the sampler. It should append the "worst" parameter onto the training set in the trainer's 
        parameter space, then return a NextStep.RunSample enum.
        

    
        -----------------------------------------------------------------------------------------------
        Arguments
        -----------------------------------------------------------------------------------------------

        trainer : Trainer
            A Trainer object that we use for training. We sample a new training point from this 
            trainer.


        -----------------------------------------------------------------------------------------------
        Returns
        -----------------------------------------------------------------------------------------------

        NextStep.RunSample : NextStep
            indicates that we have a new sample, appended it onto the train_space in the trainer's 
            parameter_space, and need to generate the FOM solution using the corresponding parameter 
            values for the IC/physics. 
        """

        raise RuntimeError("Abstract method Sampler.Sample!");



    def Generate_Training_Data(self, trainer : Trainer) -> NextStep:
        """
        This function updates trainer.U_Train and trainer.U_Test by adding solutions generated from 
        parameter combinations in trainer.param_space.train_space and trainer.param_space.test_space.

        We assume that the user has added at least one testing or training point to trainer.param_space
        which has not yet been added into trainer's U_Train or U_Test attributes. We assume that any 
        new training or testing points have been appended onto THE END of the param_space. 

        This function first determines how many testing/training parameter combinations are new (we 
        have not found the corresponding trajectories). We generate the trajectory for each of these
        parameter combinations, then append those trajectories onto trainer.U_Train/U_Test.

        If normalization is enabled, we compute a single mean/std from TRAINING data
        only (once) and normalize both training and testing trajectories using those values.


        
        -----------------------------------------------------------------------------------------------
        Arguments
        -----------------------------------------------------------------------------------------------

        trainer : Trainer
            A Trainer object that we use for training. We assume that if the user has added 
            new training parameter combinations, that they appended these new parameters onto the end 
            of trainer.param_space.train_space. Same for testing parameters.

        

        -----------------------------------------------------------------------------------------------
        Returns
        -----------------------------------------------------------------------------------------------

        NextStep.Train : NextStep
            indicates that we have generated the FOM solution for the new training point and need to 
            resume training. 
        """
        

        # ---------------------------------------------------------------------------------------------
        # Determine how many testing, training samples we need to add to U_Train/U_Test

        # Figure out how many training parameters we have not generated solution trajectories for. 
        num_train_current   : int   = len(trainer.U_Train);
        num_train_new       : int   = trainer.param_space.n_train() - num_train_current;
        assert num_train_new > 0, "num_train_new = %d <= 0" % num_train_new;
        LOGGER.info("Adding %d new parameter combinations to the training set (currently has %d)" % (num_train_new, num_train_current));

        # Fetch the parameters. We assume that if the user has added new training parameter 
        # combinations, that they appended these new parameters onto the end of param_space's 
        # train_space attribute.
        new_train_params    : numpy.ndarray         = trainer.param_space.train_space[-num_train_new:, :];
        for i in range(new_train_params.shape[0]):
            LOGGER.info("New training parameter combination %d is %s" % (i, str(new_train_params[i])));


        # Now do the same thing for testing parameters. Once again we assume that if the user added new
        # testing parameters, that they appended those parameters to the END of param_space's 
        # test_space attribute. 
        num_test_current    : int   = len(trainer.U_Test);
        num_test_new        : int   = trainer.param_space.n_test() - num_test_current;
        LOGGER.info("Adding %d new parameter combinations to the testing set (currently has %d)" % (num_test_new, num_test_current));

        if (num_test_new > 0):
            new_test_params : numpy.ndarray         = trainer.param_space.test_space[-num_test_new:, :];
            for i in range(new_test_params.shape[0]):
                LOGGER.info("new testing combination %d is %s" % (i, str(new_test_params[i])));



        # ---------------------------------------------------------------------------------------------
        # Generate new testing, training solutions.

        # Generate the FOM solutions for the new testing points. After we have generated them, we
        # append them to trainer's U_Test variable.
        if (num_test_new > 0):
            new_U_Test, new_t_Test  = trainer.physics.generate_solutions(new_test_params);

            # If normalization is already configured (stats exist), normalize the new testing data
            # before appending. If stats don't exist yet, we'll normalize the full datasets after we
            # compute stats from the training set (below).
            if hasattr(trainer, "has_normalization") and trainer.has_normalization():
                trainer.normalize_U_inplace(new_U_Test);

            if(len(trainer.U_Test) == 0):
                trainer.U_Test  : list[list[torch.Tensor]]  = new_U_Test;
                trainer.t_Test  : list[torch.Tensor]        = new_t_Test;
            else:
                trainer.U_Test : list[list[torch.Tensor]]   = trainer.U_Test + new_U_Test;
                trainer.t_Test : list[torch.Tensor]         = trainer.t_Test + new_t_Test; # type: ignore
                
            assert len(trainer.U_Test) == trainer.param_space.n_test(), "len(trainer.U_Test) = %d != trainer.param_space.n_test() = %d" % (len(trainer.U_Test), trainer.param_space.n_test());


        # Do the same thing for the training points. If a particular set of parameters is in the testing 
        # set, then we take the pre-generated solution from there rather than re-generating the solution 
        # from scratch.
        new_U_Train     : list[list[torch.Tensor]]  = [];
        new_t_Train     : list[torch.Tensor]        = [];
        
        if num_train_new > 0:
            # Vectorized search: find which training params are already in test set.
            test_space = trainer.param_space.test_space;
            test_indices_map = {};  # Map from train_idx -> test_idx for params found in test set
            missing_indices = [];   # Indices of training params that need generation
            
            for i in range(num_train_new):
                ith_train_param = new_train_params[i, :];
                # Check all test params at once: does this training param match any test param?
                diff = numpy.abs(test_space - ith_train_param)
                tol = 1e-14 + 1e-12 * numpy.abs(ith_train_param)   # atol + rtol*|b|
                is_match = numpy.all(diff <= tol, axis = 1)
                
                if numpy.any(is_match):
                    test_idx = numpy.where(is_match)[0][0];
                    LOGGER.info("Training param %d matches test param %d! Reusing solution." % (i, test_idx));
                    test_indices_map[i] = test_idx;
                else:
                    missing_indices.append(i);
            
            # Generate solutions for training params not in test set (batch call).
            generated_U_dict = {};
            generated_t_dict = {};
            if len(missing_indices) > 0:
                params_to_generate = new_train_params[missing_indices, :];
                LOGGER.info("Generating %d training solutions not found in test set." % len(missing_indices));
                generated_U_Train, generated_t_Train = trainer.physics.generate_solutions(params_to_generate);
                
                # If normalization stats exist, normalize the newly-generated training trajectories.
                if hasattr(trainer, "has_normalization") and trainer.has_normalization():
                    trainer.normalize_U_inplace(generated_U_Train);
                
                # Store generated solutions in a dict for easy lookup by original index.
                for idx, train_idx in enumerate(missing_indices):
                    generated_U_dict[train_idx] = generated_U_Train[idx];
                    generated_t_dict[train_idx] = generated_t_Train[idx];
            
            # Build final lists in the correct order (matching new_train_params order).
            for i in range(num_train_new):
                if i in test_indices_map:
                    # Copy from test set - create new list to avoid aliasing issues.
                    test_idx = test_indices_map[i];
                    new_U_Train.append([tensor.clone() for tensor in trainer.U_Test[test_idx]]);
                    new_t_Train.append(trainer.t_Test[test_idx].clone());
                    
                    # Log diagnostic info for the copied data
                    if hasattr(trainer, "has_normalization") and trainer.has_normalization():
                        test_data = trainer.U_Test[test_idx][0].reshape(-1);
                        LOGGER.debug("Copied training point from test[%d]: mean=%.6e, std=%.6e" % (
                            test_idx, float(test_data.mean().item()), float(test_data.std().item())));
                else:
                    # Use generated solution.
                    new_U_Train.append(generated_U_dict[i]);
                    new_t_Train.append(generated_t_dict[i]);

        # Now append the new training, points to U_Train.
        if(len(trainer.U_Train) == 0):
            trainer.U_Train         = new_U_Train;
            trainer.t_Train         = new_t_Train;
        else:
            # Log statistics before appending
            LOGGER.info("Before appending %d new training points:" % len(new_U_Train));
            if len(trainer.U_Train) > 0:
                old_sample = trainer.U_Train[0][0].reshape(-1);
                LOGGER.info("  Existing train[0]: mean=%.6e, std=%.6e, min=%.6e, max=%.6e" % (
                    float(old_sample.mean().item()), float(old_sample.std().item()),
                    float(old_sample.min().item()), float(old_sample.max().item())));
            if len(new_U_Train) > 0:
                new_sample = new_U_Train[0][0].reshape(-1);
                LOGGER.info("  New train[0]: mean=%.6e, std=%.6e, min=%.6e, max=%.6e" % (
                    float(new_sample.mean().item()), float(new_sample.std().item()),
                    float(new_sample.min().item()), float(new_sample.max().item())));
            
            # Initialize coefficients for newly added training points!
            # When greedy sampling adds a point from the test set, its test_coefs row may be zero/untrained.
            # We compute physics-based least-squares coefficients by encoding the trajectory and using SINDy.
            if len(new_U_Train) > 0:
                n_train_old = len(trainer.U_Train);
                LOGGER.info("Initializing coefficients for %d newly added training points using least-squares fit" % len(new_U_Train));
                
                # For each new training point, find its index in test_space and initialize its coefficients
                for i in range(len(new_U_Train)):
                    new_param = new_train_params[i];
                    
                    # Find this parameter in test_space
                    test_idx = None;
                    for j in range(trainer.param_space.n_test()):
                        if numpy.allclose(trainer.param_space.test_space[j, :], new_param, rtol=1e-12, atol=1e-14):
                            test_idx = j;
                            break;
                    
                    if test_idx is not None:
                        # New coefficients will be untrained, so we should intialize them using least-squares fit.
                        U_new_i = new_U_Train[i];  # List of tensors (one per IC)
                        t_new_i = new_t_Train[i];  # Time grid tensor
                        
                        # Move encoder_decoder to CPU for encoding (calibrate expects CPU tensors)
                        original_device = next(trainer.encoder_decoder.parameters()).device;
                        encoder_decoder_cpu = trainer.encoder_decoder.cpu();
                        with torch.no_grad():
                            # Encode trajectory 
                            Z_new_i_tuple = encoder_decoder_cpu.Encode(*[u.cpu() for u in U_new_i]);
                            Z_new_i_list = list(Z_new_i_tuple);  # Convert tuple to list
                        
                        # Move encoder_decoder back to original device
                        trainer.encoder_decoder.to(original_device);
                        
                        # Prepare inputs for calibrate: expects list[list[tensor]] where inner list has n_IC elements
                        Latent_States_list = [Z_new_i_list];  # list[list[tensor]], one param combination
                        t_Grid_list = [t_new_i.cpu()];        # list[tensor], one time grid
                        params_array = new_param.reshape(1, -1);  # Shape: (1, n_p)
                        
                        # Call calibrate with empty input_coefs to compute least-squares solution
                        output_coefs, _, _, _ = trainer.latent_dynamics.calibrate(
                                                Latent_States   = Latent_States_list,
                                                t_Grid          = t_Grid_list,
                                                input_coefs     = [],  # Empty list triggers least-squares computation
                                                loss_type       = trainer.loss_types['LD'],
                                                params          = params_array);
                        
                        # Extract the computed coefficients and assign to test_coefs
                        with torch.no_grad():
                            computed_coefs                       = output_coefs[0, :].to(trainer.device);  # Shape: (n_coefs,)
                            trainer.test_coefs[test_idx, :] = computed_coefs;
                            coef_norm                            = float(torch.norm(trainer.test_coefs[test_idx, :]).item());
                            LOGGER.info("  New training point %d (test idx %d): initialized coefficients from least-squares fit: coef norm = %.6e" % (i, test_idx, coef_norm));

            trainer.U_Train         = trainer.U_Train + new_U_Train;
            trainer.t_Train         = trainer.t_Train + new_t_Train;

        assert len(trainer.U_Train) == trainer.param_space.n_train(), "len(trainer.U_Train) = %d != trainer.param_space.n_train() = %d" % (len(trainer.U_Train), trainer.param_space.n_train());


        # ---------------------------------------------------------------------------------------------
        # Global normalization setup (training-only stats)
        #
        # If enabled and stats are not set yet, compute mean/std from the (unnormalized) training data,
        # store them on the trainer, then normalize both training and testing trajectories in-place.
        if hasattr(trainer, "normalize") and bool(trainer.normalize):
            if not (hasattr(trainer, "has_normalization") and trainer.has_normalization()):
                LOGGER.info("Computing normalization statistics...");
                # Log data stats BEFORE normalization for debugging
                sample_data = trainer.U_Train[0][0].reshape(-1);
                LOGGER.info("  Sample training point before normalization: mean=%.6e, std=%.6e, min=%.6e, max=%.6e" % (
                    float(sample_data.mean().item()), float(sample_data.std().item()),
                    float(sample_data.min().item()), float(sample_data.max().item())));
                
                # Use test set for normalization if available and training set is small
                # This gives better global statistics that work for all points in parameter space
                if len(trainer.U_Test) > 0 and len(trainer.U_Train) <= 4:
                    LOGGER.info("Using TEST set for normalization (training set is small: %d points)" % len(trainer.U_Train));
                    trainer.set_normalization_stats_from_test();
                else:
                    LOGGER.info("Using TRAINING set for normalization (%d points)" % len(trainer.U_Train));
                    trainer.set_normalization_stats_from_training();
                trainer.normalize_U_inplace(trainer.U_Train);
                trainer.normalize_U_inplace(trainer.U_Test);
                
                # Log data stats AFTER normalization for verification
                sample_data_norm = trainer.U_Train[0][0].reshape(-1);
                LOGGER.info("  After normalization: mean=%.6e, std=%.6e, min=%.6e, max=%.6e" % (
                    float(sample_data_norm.mean().item()), float(sample_data_norm.std().item()),
                    float(sample_data_norm.min().item()), float(sample_data_norm.max().item())));
            else:
                LOGGER.info("Normalization stats already exist (computed from %d initial training points)" % len(trainer.U_Train));
                LOGGER.info("  Using mean=%.6e, std=%.6e for IC 0" % (
                    float(trainer.data_mean[0].item()), float(trainer.data_std[0].item())));
                # Check if newly added data has similar statistics
                if len(trainer.U_Train) > 0:
                    last_train_data = trainer.U_Train[-1][0].reshape(-1);
                    LOGGER.info("  Last training point (normalized): mean=%.6e, std=%.6e, min=%.6e, max=%.6e" % (
                        float(last_train_data.mean().item()), float(last_train_data.std().item()),
                        float(last_train_data.min().item()), float(last_train_data.max().item())));
                    if abs(float(last_train_data.mean().item())) > 5.0:
                        LOGGER.warning("  WARNING: Normalized data has large mean! This suggests normalization stats may not be appropriate for this data!");
                    if abs(float(last_train_data.std().item()) - 1.0) > 3.0:
                        LOGGER.warning("  WARNING: Normalized data std is far from 1.0! This suggests normalization stats may not be appropriate for this data!");


        # ---------------------------------------------------------------------------------------------
        # Wrap up

        # We are now done. Since we now have the new FOM solutions, the next step is training.
        next_step = NextStep.Train;
        return next_step;

