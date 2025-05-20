# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

# Add LatentDynamics, Physics directories to the search path.
import  sys;
import  os;
LD_Path         : str = os.path.abspath(os.path.join(os.path.dirname(__file__), "LatentDynamics"));
Physics_Path    : str = os.path.abspath(os.path.join(os.path.dirname(__file__), "Physics"));
sys.path.append(LD_Path); 
sys.path.append(Physics_Path); 

import  logging;

import  numpy;
import  torch; 

import  Burgers;
import  BurgersSecondOrder;
from    LatentDynamics      import  LatentDynamics;
from    SINDy               import  SINDy;
from    DampedSpring        import  DampedSpring;
from    ParameterSpace      import  ParameterSpace;
from    GPLaSDI             import  BayesianGLaSDI;
from    Model               import  Autoencoder, load_Autoencoder, Autoencoder_Pair, load_Autoencoder_Pair;
from    Physics             import  Physics;
from    Explicit            import  Explicit;
from    NonlinearElasticity import  NonlinearElasticity

# Set up logger.
LOGGER  : logging.Logger    = logging.getLogger(__name__);

# Set up the dictionaries; we use this to allow the code to call different classes, functions 
# depending on the settings.
trainer_dict    =  {'gplasdi'               : BayesianGLaSDI};
model_dict      =  {'ae'                    : Autoencoder,
                    'pair'                  : Autoencoder_Pair};
model_load_dict =  {'ae'                    : load_Autoencoder,
                    'pair'                  : load_Autoencoder_Pair};
ld_dict         =  {'sindy'                 : SINDy, 
                    'spring'                : DampedSpring};
physics_dict    =  {'Burgers'               : Burgers.Burgers,
                    'BurgersSecondOrder'    : BurgersSecondOrder.Burgers,
                    'Explicit'              : Explicit,
                    'NonlinearElasticity'   : NonlinearElasticity};



# -------------------------------------------------------------------------------------------------
# Initialization functions
# -------------------------------------------------------------------------------------------------

def Initialize_Trainer(config : dict, restart_dict : dict = None) -> tuple[BayesianGLaSDI, ParameterSpace, Physics, torch.nn.Module, LatentDynamics]:
    """
    Initialize a LaSDI object with a latent space model and physics object according to config 
    file. Currently only 'gplasdi' is available.

    

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    config: dict
        The dictionary that we loaded from a .yml file. It should house all the settings we expect 
        to use to generate the data and train the models. We expect this dictionary to contain the 
        following keys (if a key is within a dictionary that is specified by another key, then we 
        tab the sub-key relative to the dictionary key): 
            - physics           (used by "initialize_physics")
                - type
            - latent_dynamics   (how we parameterize the latent dynamics; e.g. SINDy)
                - type
            - lasdi
                - type

    restart_dict : dict
        The dictionary returned by numpy.load when we load from a restart. This should
        contain the following keys:
            - parameter_space
            - model
            - latent_dynamics
            - trainer
            
    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    trainer, param_space, physics, model, latent_dynamics
     
    trainer : BayesianGLaSDI
        Should have been initialized using the settings in config and is ready to begin training.

    param_space : ParameterSpace
        holds the combinations of parameters in the testing and training sets.
     
    physics : Physics
        Encodes the FOM model. It allows us to fetch the FOM solution and/or initial conditions 
        for a particular combination of parameters.

    model : torch.nn.Module
        The model we use to map between the FOM and ROM spaces. Specifically, the model can 
        encode a snapshot/frame (measurement at a specific time) of the FOM solution to its 
        corresponding ROM frame. It can also decode a ROM frame back to a FOM frame. The n_IC 
        attribute of this object must match that of latent_dynamics.

    latent_dynamics : LatentDynamics
        Defines the dynamical system in model's latent space. The n_IC attribute of this object 
        must match the n_IC attribute of model.
    """

    # Fetch the trainer type. Note that only "gplasdi" is allowed.
    trainer_type            = config['lasdi']['type'];
    assert(trainer_type in config['lasdi']);
    assert(trainer_type in trainer_dict);
    LOGGER.info("Initializing Trainer (%s)" % trainer_type);

    # Set up a ParameterSpace object. This will keep track of all parameter combinations we want
    # to try during testing and training. We load the set of possible parameters and their possible
    # values using the configuration file. If we are using a restart file, then load it's 
    # ParameterSpace object.
    param_space = ParameterSpace(config);
    if (restart_dict is not None):
        param_space.load(restart_dict['parameter_space']);
    
    # Get the "physics" object we use to generate the FOM dataset.
    physics : Physics   = Initialize_Physics(config, param_space.param_names);

    # Get the model (autoencoder). We try to learn dynamics that describe how the latent space of
    # this model evolve over time. If we are using a restart file, then load the saved model 
    # parameters from file.
    if (restart_dict is not None):
        model_type : str    = config['model']['type'];
        model               = model_load_dict[model_type](restart_dict['model']);
    else: 
        model               = Initialize_Model(physics, config);

    # Initialize the latent dynamics model. If we are using a restart file, then load the saved
    # latent dynamics from this file. 
    ld_type                 = config['latent_dynamics']['type'];
    assert(ld_type in config['latent_dynamics']);
    assert(ld_type in ld_dict);
    latent_dynamics         = ld_dict[ld_type]( n_z             = model.n_z, 
                                                coef_norm_order = config['latent_dynamics']['coef_norm_order'],
                                                Uniform_t_Grid  = physics.Uniform_t_Grid);
    if (restart_dict is not None):
        latent_dynamics.load(restart_dict['latent_dynamics']);

    # Initialize the trainer object. If we are using a restart file, then load the 
    # trainer from that file.
    trainer                 = trainer_dict[trainer_type](physics, model, latent_dynamics, param_space, config['lasdi'][trainer_type]);
    if (restart_dict is not None):
        trainer.load(restart_dict['trainer']);

    # All done!
    return trainer, param_space, physics, model, latent_dynamics;



def Initialize_Model(physics : Physics, config : dict) -> torch.nn.Module:
    """
    Initialize a model (autoencoder) according to config file. 
    

    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    physics : Physics
        Encodes the FOM model. It allows us to fetch the FOM solution and/or initial conditions 
        for a particular combination of parameters. 

    config : dict
        This should be a dictionary that we loaded from a .yml file. It should house all the 
        settings we expect to use to generate the data and train the models. We expect this 
        dictionary to contain the following keys (if a key is within a dictionary that is specified 
        by another key, then we tab the sub-key relative to the dictionary key): 
            - model
                - type
    
       
    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    model : torch.nn.Module
        A torch.nn.Module object that acts as the trainable model in the gplasdi framework. This 
        model should have a latent space of some form. We learn a set of dynamics to describe how 
        this latent space evolves over time. 
    """


    # First, determine what model we are using in the latent dynamics. Make sure the user 
    # included all the information that is necessary to initialize the corresponding dynamics.
    model_type : str = config['model']['type'];
    assert(model_type in config['model']);
    assert(model_type in model_dict);
    LOGGER.info("Initializing Model (%s)" % model_type);

    # Autoencoder, autoencoder pair case.
    if(model_type == "ae" or model_type == "pair"):

        # Next, fetch the hidden widths, latent dimension (n_z), and activation for the model. 
        model_config        : dict              = config['model'][model_type];
        hidden_widths       : list[int]         = model_config['hidden_widths'];
        n_z                 : int               = model_config['latent_dimension'];
        activation          : str               = model_config['activation']  if 'activation' in model_config else 'tanh';

        # Now build the widths attribute + fetch Frame_Shape from physics.
        Frame_Shape         : list[int]         = physics.Frame_Shape;
        space_dim           : int               = numpy.prod(Frame_Shape).item();
        widths              : list[int]         = [space_dim] + hidden_widths + [n_z];

        # Now build the model!
        model           : torch.nn.Module   = model_dict[model_type](
                                                        widths          = widths, 
                                                        activation      = activation, 
                                                        reshape_shape   = Frame_Shape);

        # All done!
        return model;



def Initialize_Physics(config: dict, param_names : list[str]) -> Physics:
    '''
    Initialize a physics FOM model according to config file.

    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    config : dict
        A dictionary we loaded from a .yml file. It should house all the settings we expect to use 
        to generate the data and train the models. We expect this dictionary to contain the 
        following keys (if a key is within a dictionary that is specified by another key, then we 
        tab the sub-key relative to the dictionary key): 
            - physics 
                - type

    param_names : list[str], len  = n_p
        A list housing the names of the parameters in the physics model. There should be an entry 
        in the configuration file for each named parameter. 
            
    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    physics : Physics
        Encodes the FOM model. It allows us to fetch the FOM solution and/or initial conditions 
        for a particular combination of parameters. Initialized using the n_p parameters in the 
        config['physics'] dictionary. 
    '''

    # First, determine what kind of "physics" object we want to load.
    physics_cfg     : dict      = config['physics'];
    physics_type    : str       = physics_cfg['type'];
    LOGGER.info("Initializing Physics (%s)" % physics_type);

    # Next, initialize the "physics" object we are using to build the simulations.
    physics         : Physics   = physics_dict[physics_type](physics_cfg, param_names);

    # All done!
    return physics;
