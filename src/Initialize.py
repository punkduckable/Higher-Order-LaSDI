# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  logging;

import  numpy;
import  torch; 


from    LatentDynamics          import  LatentDynamics, SINDy, SINDy_weak, SwitchSINDy;
from    LatentDynamics          import  SwitchSINDy_weak, DampedSpring, DampedSpring_weak;

from    ParameterSpace          import  ParameterSpace;

from    Trainer                 import  Trainer, First_Order_Rollout, First_Order_Weak;
from    Trainer                 import  Second_Order_Rollout, Second_Order_Weak;

from    EncoderDecoder          import  EncoderDecoder, Autoencoder, load_Autoencoder;
from    EncoderDecoder          import  Autoencoder_Pair, load_Autoencoder_Pair;
from    EncoderDecoder          import  CNN_3D_Autoencoder, load_CNN_3D_Autoencoder;

from    Physics                 import  Physics, Burgers2D, Thermal, Burgers, BurgersSecondOrder;
from    Physics                 import  Explicit, ExplicitSecondOrder;
# from  Physics.Advection               import  Advection;             # mfem dependency.
# from  Physics.NonlinearElasticity     import  NonlinearElasticity;   # mfem dependency.
# from  Physics.WaveEquation            import  WaveEquation;          # mfem dependency.
# from  Physics.KleinGordon             import  KleinGordon;           # mfem dependency.
# from  Physics.Telegraphers            import  Telegraphers;          # mfem dependency.

from    Sample                  import  Sampler, FOM_Rollout, FOM_Variance;

# Set up logger.
LOGGER  : logging.Logger    = logging.getLogger(__name__);

# Set up the dictionaries; we use this to allow the code to call different classes, functions 
# depending on the settings.
encoder_decoder_dict = {        'ae'                        : Autoencoder,
                                'autoencoder'               : Autoencoder,
                                'pair'                      : Autoencoder_Pair,
                                'autoencoder_pair'          : Autoencoder_Pair,
                                'cnn_3d'                    : CNN_3D_Autoencoder,
                                'cnn_3d_ae'                 : CNN_3D_Autoencoder,
                                'cnn_3d_autoencoder'        : CNN_3D_Autoencoder};

encoder_decoder_load_dict = {   'ae'                        : load_Autoencoder,
                                'autoencoder'               : load_Autoencoder,
                                'pair'                      : load_Autoencoder_Pair,
                                'autoencoder_pair'          : load_Autoencoder_Pair,
                                'cnn_3d'                    : load_CNN_3D_Autoencoder,
                                'cnn_3d_ae'                 : load_CNN_3D_Autoencoder,
                                'cnn_3d_autoencoder'        : load_CNN_3D_Autoencoder};

ld_dict = {                     'sindy'                     : SINDy,
                                'sindy_w'                   : SINDy_weak,
                                'spring'                    : DampedSpring,
                                'spring_w'                  : DampedSpring_weak,
                                'switch'                    : SwitchSINDy,
                                'switch_w'                  : SwitchSINDy_weak};

trainer_dict = {                'First_Order_Rollout'       : First_Order_Rollout,
                                'First_Order_Weak'          : First_Order_Weak,
                                'Second_Order_Rollout'      : Second_Order_Rollout,
                                'Second_Order_Weak'         : Second_Order_Weak};

sampler_dict = {                'FOM_Rollout'               : FOM_Rollout,
                                'FOM_Variance'              : FOM_Variance};

physics_dict = {                'Burgers'                   : Burgers.Burgers,
                                'BurgersSecondOrder'        : BurgersSecondOrder.Burgers,
                                'Burgers2D'                 : Burgers2D.Burgers2D,
                                'Explicit'                  : Explicit.Explicit,
                                'ExplicitSecondOrder'       : ExplicitSecondOrder.Explicit,
                                'Thermal'                   : Thermal.Thermal,
                                # 'Advection'                 : Advection.Advection,                           # mfem dependency
                                # 'NonlinearElasticity'       : NonlinearElasticity.NonlinearElasticity,       # mfem dependency
                                # 'WaveEquation'              : WaveEquation.WaveEquation,                      # mfem dependency
                                # 'KleinGordon'               : KleinGordon.KleinGordon,                        # mfem dependency
                                # 'Telegraphers'              : Telegraphers.Telegraphers                       # mfem dependency
                                };



# -------------------------------------------------------------------------------------------------
# Initialization functions
# -------------------------------------------------------------------------------------------------

def Initialize_Trainer( 
        config                  : dict, 
        restart_dict            : dict  = {},
        make_restart_checkpoint : bool  = True,
    ) -> tuple[Trainer, Sampler, ParameterSpace, Physics, EncoderDecoder, LatentDynamics]:
    """
    Initialize a Trainer object with a latent space model and physics object according to config 
    file. 

    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    config: dict
        The dictionary that we loaded from a .yml file. It should house all the settings we expect 
        to use to generate the data and train the encoder_decoders. We expect this dictionary to 
        contain the following keys (if a key is within a dictionary that is specified by another key, 
        then we tab the sub-key relative to the dictionary key): 
            - physics           (used by "initialize_physics")
                - type
            - latent_dynamics   (how we parameterize the latent dynamics; e.g. SINDy)
                - type
            - trainer

    restart_dict : dict, optional
        If provided, then we will use the settings in this dictionary to initialize the trainer, 
        parameter space, physics, encoder_decoder, and latent dynamics. If not provided, then we will 
        initialize everything from scratch.

    make_restart_checkpoint : bool, optional
        If True and restart_dict is provided, then make a checkpoint using the loaded 
        encoder_decoder parameters. This preserves the original restart behavior for training. Set
        this to False when loading a saved artifact for analysis only.
            
    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    trainer, sampler, param_space, physics, encoder_decoder, latent_dynamics
     
    trainer : Trainer
        Should have been initialized using the settings in config and is ready to begin training.

    sampler : Sampler
        The sampler object used to select the "worst" testing parameter combination during greedy 
        sampling.

    param_space : ParameterSpace
        holds the combinations of parameters in the testing and training sets.
     
    physics : Physics
        Encodes the FOM model. It allows us to fetch the FOM solution and/or initial conditions 
        for a particular combination of parameters.

    encoder_decoder : EncoderDecoder
        The encoder_decoder we use to map between the FOM and ROM spaces. Specifically, the 
        encoder_decoder can encode a snapshot/frame (measurement at a specific time) of the FOM 
        solution to its corresponding ROM frame. It can also decode a ROM frame back to a FOM
        frame. The n_IC attribute of this object must match that of latent_dynamics.

    latent_dynamics : LatentDynamics
        Defines the dynamical system in encoder_decoder's latent space. The n_IC attribute of this 
        object must match the n_IC attribute of encoder_decoder.
    """

    # Set up a ParameterSpace object. This will keep track of all parameter combinations we want
    # to try during testing and training. We load the set of possible parameters and their possible
    # values using the configuration file. If we are using a restart file, then load it's 
    # ParameterSpace object.
    param_space = ParameterSpace(config);
    if (bool(restart_dict) == True):        # Empty dictionaries evaluate to False. restart_dict is empty if we are not using a restart file.
        param_space.load(restart_dict['parameter_space']);
    
    # Get the "physics" object we use to generate the FOM dataset.
    physics : Physics   = Initialize_Physics(config, param_space.param_names);
    if (bool(restart_dict) == True):        # Empty dictionaries evaluate to False. restart_dict is empty if we are not using a restart file.
        physics.load(restart_dict['physics']);

    # Get the encoder_decoder. We try to learn dynamics that describe how the latent space of
    # this encoder_decoder evolve over time. If we are using a restart file, then load the saved 
    # encoder_decoder parameters from file.
    if (bool(restart_dict) == True):        # Empty dictionaries evaluate to False. restart_dict is empty if we are not using a restart file.
        encoder_decoder_type : str    = config['EncoderDecoder']['type'];
        encoder_decoder               = encoder_decoder_load_dict[encoder_decoder_type](restart_dict['encoder_decoder'], config['EncoderDecoder']);
    else: 
        encoder_decoder               = Initialize_Encoder_Decoder(physics, config);

    # Initialize the latent dynamics model. If we are using a restart file, then load the saved
    # latent dynamics from this file. 
    ld_type                 = config['latent_dynamics']['type'];
    assert(ld_type in config['latent_dynamics']);
    assert(ld_type in ld_dict);
    if(ld_type == "switch" or ld_type == "switch_w"):
        latent_dynamics         = ld_dict[ld_type]( n_z             = encoder_decoder.n_z, 
                                                    Uniform_t_Grid  = physics.Uniform_t_Grid,
                                                    switch_time     = physics.switch_time,
                                                    config          = config['latent_dynamics']);
    else:
        latent_dynamics         = ld_dict[ld_type]( n_z             = encoder_decoder.n_z, 
                                                    Uniform_t_Grid  = physics.Uniform_t_Grid,
                                                    config          = config['latent_dynamics']);
    
    if (bool(restart_dict) == True):        # Empty dictionaries evaluate to False. restart_dict is empty if we are not using a restart file.
        latent_dynamics.load(restart_dict['latent_dynamics']);

    # Initialize the trainer object. If we are using a restart file, then load the 
    # trainer from that file.
    trainer_type            = config['trainer']['type'];
    trainer                 = trainer_dict[trainer_type](physics, encoder_decoder, latent_dynamics, param_space, config);
    
    if (bool(restart_dict) == True):        # Empty dictionaries evaluate to False. restart_dict is empty if we are not using a restart file.
        trainer.load(restart_dict['trainer']);

    # Check if we should make a checkpoint using the current encoder_decoder parameters.
    if (bool(restart_dict) == True and make_restart_checkpoint == True): 
        trainer._Save_Checkpoint(   encoder_decoder = encoder_decoder, 
                                    iter            = trainer.restart_iter);

    # Load the sampler.
    sampler_type    : str       = config['sampler']['type'];
    sampler         : Sampler   = sampler_dict[sampler_type](config['sampler']);
    
    # All done!
    return trainer, sampler, param_space, physics, encoder_decoder, latent_dynamics;



def Initialize_Encoder_Decoder(physics : Physics, config : dict) -> EncoderDecoder:
    """
    Initialize a encoder_decoder (autoencoder) according to config file. 
    

    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    physics : Physics
        Encodes the FOM model. It allows us to fetch the FOM solution and/or initial conditions 
        for a particular combination of parameters. 

    config : dict
        This should be a dictionary that we loaded from a .yml file. It should house all the 
        settings we expect to use to generate the data and train the encoder_decoder. We expect 
        this dictionary to contain the following keys (if a key is within a dictionary that is 
        specified by another key, then we tab the sub-key relative to the dictionary key): 
            - encoder_decoder
                - type
    
       
    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    encoder_decoder : EncoderDecoder
        A EncoderDecoder object that acts as the trainable encoder_decoder in the LaSDI framework. 
        This encoder_decoder should have a latent space of some form. We learn a set of dynamics to 
        describe how this latent space evolves over time. 
    """
    
    # First, determine what encoder_decoder we are using in the latent dynamics. Make sure the user 
    # included all the information that is necessary to initialize the corresponding dynamics.
    encoder_decoder_type : str = config['EncoderDecoder']['type'];
    assert(encoder_decoder_type in config['EncoderDecoder']);
    assert(encoder_decoder_type in encoder_decoder_dict);
    LOGGER.info("Initializing EncoderDecoder (%s)" % encoder_decoder_type);

    encoder_decoder = encoder_decoder_dict[encoder_decoder_type]( 
                                                Frame_Shape = physics.Frame_Shape,
                                                config      = config['EncoderDecoder']);

    return encoder_decoder;



def Initialize_Physics(config: dict, param_names : list[str]) -> Physics:
    '''
    Initialize a physics FOM model according to config file.

    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    config : dict
        A dictionary we loaded from a .yml file. It should house all the settings we expect to use 
        to generate the data and train the encoder_decoders. We expect this dictionary to contain 
        the following keys (if a key is within a dictionary that is specified by another key, then 
        we tab the sub-key relative to the dictionary key): 
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
