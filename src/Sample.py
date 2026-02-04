# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  logging;

import  torch;
import  numpy;

from    Enums               import  NextStep, Result;  
from    GPLaSDI             import  BayesianGLaSDI;


# Setup logger.
LOGGER : logging.Logger = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# Sampling functions
# -------------------------------------------------------------------------------------------------

def Update_Train_Space(trainer : BayesianGLaSDI) -> tuple[NextStep, Result]:
    """
    This function uses greedy sampling to update the trainer's train_space.

    

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    trainer : BayesianGLaSDI
        A BayesianGLaSDI object that we use for training. We sample a new training point 
        from this trainer.


    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    NextStep.RunSample, Result.Success

    NextStep.RunSample : NextStep
        indicates that we have a new sample and need to generate the FOM solution using the 
        corresponding parameter values for the IC/physics. 
    
    Result.Success : Result 
        indicates that we were able to pick a new sample without running into any problems. 
    """

    # Figure out if we need a new sample.
    #
    # If this is the first step, trainer.U_Train will be empty, meaning that we need to run a
    # simulation for every combination of parameters in the train_space.
    # 
    # By contrast, if this is not the initial step, we need to use greedy sampling to pick a new
    # combination of parameter values, then append it to the train space.
    if(len(trainer.U_Train) != 0):
        new_sample  : numpy.ndarray = trainer.get_new_sample_point();
        trainer.param_space.appendTrainSpace(new_sample);

    # Now that we know the new points we need to generate simulations for, we need to get ready to
    # actually run those simulations.
    next_step, result = NextStep.RunSample, Result.Success;
    return result, next_step;



def Run_Samples(trainer : BayesianGLaSDI) -> tuple[NextStep, Result]:
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

    trainer : BayesianGLaSDI
        A BayesianGLaSDI object that we use for training. We assume that if the user has added 
        new training parameter combinations, that they appended these new parameters onto the end 
        of trainer.param_space.train_space. Same for testing parameters.

    

    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    NextStep.Train, Result.Success

    NextStep.Train : NextStep
        indicates that we have generated the FOM solution for the new training point and need to 
        resume training. 
    
    Result.Success : Result 
        indicates that we were able to pick a new sample without running into any problems. 
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
            LOGGER.info("new training combination %d is %s" % (i, str(new_test_params[i])));


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
            trainer.t_Test : list[torch.Tensor]         = trainer.t_Test + new_t_Test;
            
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
            is_match = numpy.all(test_space == ith_train_param, axis=1);
            
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
                # Copy from test set.
                new_U_Train.append(trainer.U_Test[test_indices_map[i]]);
                new_t_Train.append(trainer.t_Test[test_indices_map[i]]);
            else:
                # Use generated solution.
                new_U_Train.append(generated_U_dict[i]);
                new_t_Train.append(generated_t_dict[i]);

    # Now append the new training, points to U_Train.
    if(len(trainer.U_Train) == 0):
        trainer.U_Train         = new_U_Train;
        trainer.t_Train         = new_t_Train;
    else:
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
            trainer.set_normalization_stats_from_training();
            trainer.normalize_U_inplace(trainer.U_Train);
            trainer.normalize_U_inplace(trainer.U_Test);


    # ---------------------------------------------------------------------------------------------
    # Wrap up

    # We are now done. Since we now have the new FOM solutions, the next step is training.
    next_step, result = NextStep.Train, Result.Success;
    return result, next_step;