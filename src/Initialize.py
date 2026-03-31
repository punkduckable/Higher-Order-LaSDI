# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

# Add LatentDynamics, Physics, and EncoderDecoder directories to the search path.
import  sys;
import  os;
LD_Path             : str = os.path.abspath(os.path.join(os.path.dirname(__file__), "LatentDynamics"));
Physics_Path        : str = os.path.abspath(os.path.join(os.path.dirname(__file__), "Physics"));
EncoderDecoder_Path : str = os.path.abspath(os.path.join(os.path.dirname(__file__), "EncoderDecoder"));
sys.path.append(LD_Path); 
sys.path.append(Physics_Path); 
sys.path.append(EncoderDecoder_Path); 

import  logging;

import  numpy;
import  torch; 


from    LatentDynamics      import  LatentDynamics;
from    SINDy               import  SINDy;
from    SwitchSINDy         import  SwitchSINDy;
from    DampedSpring        import  DampedSpring;

from    ParameterSpace      import  ParameterSpace;
from    Trainer             import  Trainer;

from    EncoderDecoder      import  EncoderDecoder;
from    Autoencoder         import  Autoencoder, load_Autoencoder;
from    Autoencoder_Pair    import  Autoencoder_Pair, load_Autoencoder_Pair;
from    CNN_3D_Autoencoder  import  CNN_3D_Autoencoder, load_CNN_3D_Autoencoder;

from    Physics             import  Physics;
#from    NonlinearElasticity import  NonlinearElasticity;
#from    Advection           import  Advection;
#from    WaveEquation        import  WaveEquation;
#from    KleinGordon         import  KleinGordon;
#from    Telegraphers        import  Telegraphers;
import  Burgers2D;
import  Thermal;
import  Burgers;
import  BurgersSecondOrder;
import  Explicit;
import  ExplicitSecondOrder;

from    Sampler             import  Sampler;
from    FOM_Rollout         import  FOM_Rollout;
from    FOM_Variance        import  FOM_Variance;

# Set up logger.
LOGGER  : logging.Logger    = logging.getLogger(__name__);

# Set up the dictionaries; we use this to allow the code to call different classes, functions 
# depending on the settings.
encoder_decoder_dict = {        'ae'                    : Autoencoder,
                                'autoencoder'           : Autoencoder,
                                'pair'                  : Autoencoder_Pair,
                                'autoencoder_pair'      : Autoencoder_Pair,
                                'cnn_3d'                : CNN_3D_Autoencoder,
                                'cnn_3d_ae'             : CNN_3D_Autoencoder,
                                'cnn_3d_autoencoder'    : CNN_3D_Autoencoder};
encoder_decoder_load_dict = {   'ae'                    : load_Autoencoder,
                                'autoencoder'           : load_Autoencoder,
                                'pair'                  : load_Autoencoder_Pair,
                                'autoencoder_pair'      : load_Autoencoder_Pair,
                                'cnn_3d'                : load_CNN_3D_Autoencoder,
                                'cnn_3d_ae'             : load_CNN_3D_Autoencoder,
                                'cnn_3d_autoencoder'    : load_CNN_3D_Autoencoder};
ld_dict = {                     'sindy'                 : SINDy, 
                                'spring'                : DampedSpring,
                                'switch'                : SwitchSINDy};
sampler_dict = {                'FOM_Rollout'           : FOM_Rollout,
                                'FOM_Variance'          : FOM_Variance};
physics_dict = {                'Burgers'               : Burgers.Burgers,
                                'BurgersSecondOrder'    : BurgersSecondOrder.Burgers,
                                'Burgers2D'             : Burgers2D.Burgers2D,
                                'Explicit'              : Explicit.Explicit,
                                'ExplicitSecondOrder'   : ExplicitSecondOrder.Explicit,
                                'Thermal'               : Thermal.Thermal};
"""
                                'Advection'             : Advection,
                                'NonlinearElasticity'   : NonlinearElasticity,
                                'WaveEquation'          : WaveEquation,
                                'KleinGordon'           : KleinGordon,
                                'Telegraphers'          : Telegraphers};
"""



# -------------------------------------------------------------------------------------------------
# Initialization functions
# -------------------------------------------------------------------------------------------------

def Initialize_Trainer(config : dict, restart_dict : dict = {}) -> tuple[Trainer, Sampler, ParameterSpace, Physics, EncoderDecoder, LatentDynamics]:
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
        encoder_decoder               = encoder_decoder_load_dict[encoder_decoder_type](restart_dict['encoder_decoder']);
    else: 
        encoder_decoder               = Initialize_Encoder_Decoder(physics, config);

    # Initialize the latent dynamics model. If we are using a restart file, then load the saved
    # latent dynamics from this file. 
    ld_type                 = config['latent_dynamics']['type'];
    assert(ld_type in config['latent_dynamics']);
    assert(ld_type in ld_dict);
    if(ld_type == "switch"):
        latent_dynamics         = ld_dict[ld_type]( n_z             = encoder_decoder.n_z, 
                                                    Uniform_t_Grid  = physics.Uniform_t_Grid,
                                                    switch_time     = physics.switch_time,
                                                    lstsq_reg       = config['latent_dynamics'].get('lstsq_reg', 1.0));
    else:
        latent_dynamics         = ld_dict[ld_type]( n_z             = encoder_decoder.n_z, 
                                                    Uniform_t_Grid  = physics.Uniform_t_Grid,
                                                    lstsq_reg       = config['latent_dynamics'].get('lstsq_reg', 1.0));
    
    if (bool(restart_dict) == True):        # Empty dictionaries evaluate to False. restart_dict is empty if we are not using a restart file.
        latent_dynamics.load(restart_dict['latent_dynamics']);

    # Initialize the trainer object. If we are using a restart file, then load the 
    # trainer from that file.
    trainer                 = Trainer(physics, encoder_decoder, latent_dynamics, param_space, config);
    if (bool(restart_dict) == True):        # Empty dictionaries evaluate to False. restart_dict is empty if we are not using a restart file.
        trainer.load(restart_dict['trainer']);

    # If we are loading from a restart file, make a checkpoint using the current encoder_decoder parameters.
    if (bool(restart_dict) == True): 
        torch.save(encoder_decoder.cpu().state_dict(), trainer.path_checkpoint + '/' + 'checkpoint.pt');

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

    # Autoencoder, autoencoder pair case.
    if(encoder_decoder_type == "ae" or encoder_decoder_type == "pair" or encoder_decoder_type == "autoencoder" or encoder_decoder_type == "autoencoder_pair"):
        # Next, fetch the hidden widths and latent dimension (n_z). 
        encoder_decoder_config  : dict              = config['EncoderDecoder'][encoder_decoder_type];
        hidden_widths           : list[int]         = encoder_decoder_config['hidden_widths'];
        n_z                     : int               = encoder_decoder_config['latent_dimension'];

        # Fetch the activations. This can either be a string or a list of strings. If it's 
        # a string, then we use that activation for all layers.
        n_hidden_layers     : int               = len(hidden_widths);
        if(isinstance(encoder_decoder_config['activations'], str)):
            activations         : list[str]     = [encoder_decoder_config['activations']] * n_hidden_layers;   # The final layer has no activation.
        elif(isinstance(encoder_decoder_config['activations'], list)):
            activations         : list[str]     = encoder_decoder_config['activations'];
            assert(len(activations) == n_hidden_layers);
        else:
            raise ValueError("Activations must be a string or a list of strings.");

        # Now build the widths attribute + fetch Frame_Shape from physics.
        Frame_Shape         : list[int]         = physics.Frame_Shape;
        space_dim           : int               = numpy.prod(Frame_Shape).item();
        widths              : list[int]         = [space_dim] + hidden_widths + [n_z];

        # Now build the encoder_decoder!
        encoder_decoder : EncoderDecoder        = encoder_decoder_dict[encoder_decoder_type](
                                                        widths          = widths, 
                                                        activations     = activations, 
                                                        reshape_shape   = Frame_Shape);

        # All done!
        return encoder_decoder;


    # Convolutional autoencoder case.
    elif(encoder_decoder_type == "cnn_3d" or encoder_decoder_type == "cnn_3d_ae" or encoder_decoder_type == "cnn_3d_autoencoder"):
        encoder_decoder_config  : dict              = config['EncoderDecoder'][encoder_decoder_type];

        # FC configuration (analogous to the AE's hidden_widths/activations).
        hidden_widths_fc        : list[int]         = encoder_decoder_config.get('hidden_widths_fc', encoder_decoder_config.get('hidden_widths'));
        n_z                     : int               = encoder_decoder_config['latent_dimension'];

        # FC activations can either be a string or a list of strings.
        n_hidden_layers         : int               = len(hidden_widths_fc);
        act_cfg = encoder_decoder_config.get('activations_fc', encoder_decoder_config.get('activations'));
        if(isinstance(act_cfg, str)):
            activations_fc      : list[str]        = [act_cfg] * n_hidden_layers;
        elif(isinstance(act_cfg, list)):
            activations_fc      : list[str]        = act_cfg;
            assert(len(activations_fc) == n_hidden_layers);
        else:
            raise ValueError("activations_fc must be a string or a list of strings.");

        # Conv configuration.
        conv_channels       : list[int]         = encoder_decoder_config['conv_channels'];
        conv_kernel_sizes   = encoder_decoder_config.get('conv_kernel_sizes', 3);
        conv_strides        = encoder_decoder_config.get('conv_strides', 2);
        conv_paddings       = encoder_decoder_config.get('conv_paddings', 1);

        # Per-layer conv activations. This can be a string (use same activation for all conv layers)
        # or a list of strings of length len(conv_channels) - 1.
        conv_act_cfg = encoder_decoder_config.get('conv_activations', 'relu');
        if(isinstance(conv_act_cfg, str)):
            conv_activations : list[str] = [conv_act_cfg] * (len(conv_channels) - 1);
        elif(isinstance(conv_act_cfg, list)):
            conv_activations = conv_act_cfg;
            assert(len(conv_activations) == len(conv_channels) - 1);
        else:
            raise ValueError("conv_activations must be a string or a list of strings.");

        # Fetch Frame_Shape from physics (must be 3D for Conv3d).
        Frame_Shape         : list[int]         = physics.Frame_Shape;
        assert(len(Frame_Shape) == 4), "physics.Frame_Shape = %s; Conv_Autoencoder requires a 3D spatial shape" % str(Frame_Shape);
        C               : int       = int(Frame_Shape[0]);
        reshape_shape   : list[int] = [int(x) for x in Frame_Shape[1:]];
        assert conv_channels[0] == C, "conv_chanels[0] = %d, but the data has %d channels. These must match" % (conv_channels[0], C);

        encoder_decoder     : EncoderDecoder    = encoder_decoder_dict[encoder_decoder_type](
                                                        reshape_shape        = reshape_shape,
                                                        hidden_widths_fc     = hidden_widths_fc,
                                                        activations_fc       = activations_fc,
                                                        latent_dimension     = n_z,
                                                        conv_channels        = conv_channels,
                                                        conv_kernel_sizes    = conv_kernel_sizes,
                                                        conv_strides         = conv_strides,
                                                        conv_paddings        = conv_paddings,
                                                        conv_activations     = conv_activations);

        return encoder_decoder;

    else:
        raise ValueError("EncoderDecoder type %s not supported." % encoder_decoder_type);



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
