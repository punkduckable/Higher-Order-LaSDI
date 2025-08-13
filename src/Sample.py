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

def Update_Train_Space(trainer : BayesianGLaSDI, config : dict) -> tuple[NextStep, Result]:
    """
    This function uses greedy sampling to update the trainer's train_space.

    

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    trainer : BayesianGLaSDI
        A BayesianGLaSDI object that we use for training. We sample a new training point 
        from this trainer.

    config : dict
        This should be a dictionary that we loaded from a .yml file. It should house all the 
        settings we expect to use to generate the data and train the models.
    

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



def Run_Samples(trainer : BayesianGLaSDI, config : dict) -> tuple[NextStep, Result]:
    """
    This function updates trainer.U_Train and trainer.U_Test by adding solutions generated from 
    parameter combinations in trainer.param_space.train_space and trainer.param_space.test_space.

    We assume that the user has added at least one testing or training point to trainer.param_space
    which has not yet been added into trainer's U_Train or U_Test attributes. We assume that any 
    new training or testing points have been appended onto THE END of the param_space. 

    This function first determines how many testing/training parameter combinations are new (we 
    have not found the corresponding trajectories). We generate the trajectory for each of these
    parameter combinations, then append those trajectories onto trainer.U_Train/U_Test. It also 
    computes the std of the FOM solution for each combination of training parameters and stores 
    the results in trainer.std_Train.


    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    trainer : BayesianGLaSDI
        A BayesianGLaSDI object that we use for training. 

    config : dict
        This should be a dictionary that we loaded from a .yml file. It should house all the 
        settings we expect to use to generate the data and train the models.

    

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
    if(len(trainer.U_Train) == 0):
        num_train_current   : int   = 0;
        num_train_new       : int   = trainer.param_space.n_train();
    else:
        num_train_current   : int   = len(trainer.U_Train);
        num_train_new       : int   = trainer.param_space.n_train() - num_train_current;
    assert(num_train_new > 0);
    LOGGER.info("Adding %d new parameter combinations to the training set (currently has %d)" % (num_train_new, num_train_current));

    # Fetch the parameters. We assume that if the user has added new training parameter 
    # combinations, that they appended these new parameters onto the end of param_space's 
    # train_space attribute.
    new_train_params    : numpy.ndarray         = trainer.param_space.train_space[-num_train_new:, :];
    for i in range(new_train_params.shape[0]):
        LOGGER.debug("new training combination %d is %s" % (i, str(new_train_params[i])));


    # Now do the same thing for testing parameters. Once again we assume that if the user added new
    # testing parameters, that they appended those parameters to the END of param_space's 
    # test_space attribute. 
    if(len(trainer.U_Test) == 0):
        num_test_current    : int   = 0;
        num_test_new        : int   = trainer.param_space.n_test();
    else:
        num_test_current    : int   = len(trainer.U_Test);
        num_test_new        : int   = trainer.param_space.n_test() - num_test_current;
    LOGGER.info("Adding %d new parameter combinations to the testing set (currently has %d)" % (num_test_new, num_test_current));

    if (num_test_new > 0):
        new_test_params : numpy.ndarray         = trainer.param_space.test_space[-num_test_new:, :];
        for i in range(new_test_params.shape[0]):
            LOGGER.debug("new training combination %d is %s" % (i, str(new_test_params[i])));


    # ---------------------------------------------------------------------------------------------
    # Generate new testing, training solutions.

    # Generate the FOM solutions for the new testing points. After we have generated them, we
    # append them to trainer's U_Test variable.
    if (num_test_new > 0):
        new_U_Test, new_t_Test  = trainer.physics.generate_solutions(new_test_params);

        if(len(trainer.U_Test) == 0):
            trainer.U_Test  : list[list[torch.Tensor]]  = new_U_Test;
            trainer.t_Test  : list[torch.Tensor]        = new_t_Test;
        else:
            trainer.U_Test : list[list[torch.Tensor]]   = trainer.U_Test + new_U_Test;
            trainer.t_Test : list[torch.Tensor]         = trainer.t_Test + new_t_Test;
            
        assert(len(trainer.U_Test) == trainer.param_space.n_test());


    # Do the same thing for the training points. We do this one at a time. If a particular set of
    # parameters is in the testing set, then we take the pre-generated solution from there rather
    # than re-generating the solution from scratch. Also, we compute the std of the FOM solution 
    # for each combination of training parameters and store the results in trainer.std_Train.
    new_U_Train     : list[list[torch.Tensor]]  = [];
    new_t_Train     : list[torch.Tensor]        = [];
    new_std_Train   : list[list[float]]         = [];
    for i in range(num_train_new):
        # Check if the i'th combination of training parameters is in the testing set.
        ith_Train_param     : numpy.ndarray = new_train_params[i, :];
        n_test              : int           = trainer.param_space.n_test();
        found_param_in_test : bool          = False;
        LOGGER.info("Searching for training parameter %d, %s, in the testing set." % (i, str(ith_Train_param)));
        for j in range(n_test):
            # Check if every element of the j'th element of test_space matches every element of 
            # the i'th new training param. If so, copy the solution and t_Grid from trainer.
            if(numpy.any(numpy.all(trainer.param_space.test_space[j, :] == ith_Train_param))):
                LOGGER.info("Train parameter %d is test parameter %d! Copying the solution and t_grid from the testing set!" % (i, j));
                new_U_Train.append(trainer.U_Test[j]);
                new_t_Train.append(trainer.t_Test[j]);
                found_param_in_test = True;
                break;
        
        # If we could not find the training parameter, then we need to generate it.
        if(found_param_in_test == False):
            LOGGER.info("Couldn't find training parameter %d in the testing set; generating solution" % i);
            ith_new_U_Train, ith_new_t_Train = trainer.physics.generate_solutions(new_train_params[i, :].reshape(1, -1));
            new_U_Train = new_U_Train + ith_new_U_Train;
            new_t_Train = new_t_Train + ith_new_t_Train;

        # compute the std of the FOM solution for the i'th combination of training parameters.
        jth_std_Train : list[float] = [];
        for k in range(trainer.n_IC):
            jth_std_Train.append(torch.std(new_U_Train[-1][k]).item());
        new_std_Train.append(jth_std_Train); 
    
    # Now append the new training, points to U_Train.
    if(len(trainer.U_Train) == 0):
        trainer.U_Train         = new_U_Train;
        trainer.t_Train         = new_t_Train;
        trainer.std_Train       = new_std_Train;
    else:
        trainer.U_Train         = trainer.U_Train + new_U_Train;
        trainer.t_Train         = trainer.t_Train + new_t_Train;
        trainer.std_Train       = trainer.std_Train + new_std_Train;

    assert(len(trainer.U_Train) == trainer.param_space.n_train());


    # ---------------------------------------------------------------------------------------------
    # Wrap up

    # We are now done. Since we now have the new FOM solutions, the next step is training.
    next_step, result = NextStep.Train, Result.Success;
    return result, next_step;