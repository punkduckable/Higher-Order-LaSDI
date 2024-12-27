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

def Pick_Samples(trainer : BayesianGLaSDI, config : dict) -> tuple[NextStep, Result]:
    """
    This function uses greedy sampling to pick a new parameter point.

    

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    trainer: A BayesianGLaSDI object that we use for training. We sample a new training point 
    from this trainer.

    config: This should be a dictionary that we loaded from a .yml file. It should house all the 
    settings we expect to use to generate the data and train the models.
    

    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    A tuple: (NextStep.RunSample, Result.Success). The first returned value, NextStep.RunSample, 
    indicates that we have a new sample and need to generate the fom solution using the 
    corresponding parameter values for the IC. The second returned value, Result.Success, indicates 
    that we were able to pick a new sample without running into any problems. 
    """

    # First, figure out which samples we need to run simulations for. 
    if(trainer.X_Train[0].shape[0] == 0):
        # If this is the initial step then trainer.X_Train will be empty, meaning that we need to 
        # run a simulation for every combination of parameters in the train_space. 
        new_sample  : numpy.ndarray = trainer.param_space.train_space;
    else:
        # If this is not the initial step, then we need to use greedy sampling to pick the new 
        # combination of parameter values.
        new_sample  : numpy.ndarray = trainer.get_new_sample_point();
        trainer.param_space.appendTrainSpace(new_sample);

    # Now that we know the new points we need to generate simulations for, we need to get ready to
    # actually run those simulations.
    next_step, result = NextStep.RunSample, Result.Success;
    return result, next_step;



def Run_Samples(trainer : BayesianGLaSDI, config : dict) -> tuple[NextStep, Result]:
    """
    This function updates trainer.X_Train and trainer.X_Test based on param_space.train_space and 
    param_space.test_space.


    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    trainer: A BayesianGLaSDI object that we use for training. 

    config: This should be a dictionary that we loaded from a .yml file. It should house all the 
    settings we expect to use to generate the data and train the models.

    

    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------


    A tuple: (NextStep.Train, Result.Success). The first returned value, NextStep.Train, 
    indicates that we have generated the fom solution for the new training point and need to 
    resume training. The second return value, Result.Success, indicates that we were able to 
    generate the fom solution without running into any problems. 
    """
    

    # ---------------------------------------------------------------------------------------------
    # Determine how many testing, training samples we need to add

    # Figure out how many new training examples there are. Note: we require there is at least one.
    if(trainer.X_Train[0].shape[0] == 0):
        new_trains      : int                   = trainer.param_space.n_train();
    else:
        new_trains      : int                   = trainer.param_space.n_train() - trainer.X_Train[0].size(0);
    assert(new_trains > 0);
    LOGGER.info("Adding %d new parameter combinations to the training set (currently has %d)" % (new_trains, trainer.X_Train[0].shape[0]));

    # Fetch the parameters. The i'th row of this matrix gives the i'th combination of parameter
    # values for which we have not generated a fom solution.
    new_train_params    : numpy.ndarray         = trainer.param_space.train_space[-new_trains:, :]
    for i in range(new_train_params.shape[0]):
        LOGGER.debug("new training combination %d is %s" % (i, str(new_train_params[i])));

    # Figure out how many new testing parameter combinations there are. If there are any, fetch 
    # them from the param space.
    if(trainer.X_Test[0].shape[0] == 0):
        new_tests       : int                   = trainer.param_space.n_test();
    else:
        new_tests       : int                   = trainer.param_space.n_test() - trainer.X_Test[0].size(0)
    LOGGER.info("Adding %d new parameter combinations to the testing set (currently has %d)" % (new_tests, trainer.X_Test[0].size(0)));

    if (new_tests > 0):
        new_test_params : numpy.ndarray         = trainer.param_space.test_space[-new_tests:, :]
        for i in range(new_test_params.shape[0]):
            LOGGER.debug("new training combination %d is %s" % (i, str(new_test_params[i])));


    # ---------------------------------------------------------------------------------------------
    # Generate new testing, training solutions.

    # Generate the fom solutions for the new training points. After we have generated them, we
    # append them to trainer's X_Train variable.
    new_X               : list[torch.Tensor]    = trainer.physics.generate_solutions(new_train_params);
    if(trainer.X_Train[0].shape[0] == 0):
        trainer.X_Train     = new_X;
    else:
        assert(len(new_X) == len(trainer.X_Train));
        for i in range(len(new_X)):
            trainer.X_Train[i]                  = torch.cat([trainer.X_Train[i], new_X[i]], dim = 0)

    assert(trainer.X_Train[0].shape[0] == trainer.param_space.n_train())

    
    # Do the same thing for the testing points.
    if (new_tests > 0):
        new_X           : list[torch.Tensor]    = trainer.physics.generate_solutions(new_test_params);

        if(trainer.X_Test[0].shape[0] == 0):
            trainer.X_Test = new_X;
        else:
            assert(len(new_X) == len(trainer.X_Test));
            for i in range(len(new_X)):
                trainer.X_Test[i]               = torch.cat([trainer.X_Test[i], new_X[i]], dim = 0)
            
        assert(trainer.X_Test[0].size(0) == trainer.param_space.n_test())


    # ---------------------------------------------------------------------------------------------
    # Wrap up

    # We are now done. Since we now have the new fom solutions, the next step is training.
    next_step, result = NextStep.Train, Result.Success
    return result, next_step