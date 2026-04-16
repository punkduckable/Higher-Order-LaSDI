# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

# Add the Physics directory to the search path.
import  sys;
import  os;
src_Path        : str   = os.path.abspath(os.path.dirname(os.path.dirname(__file__)));
Physics_Path    : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, "Physics"));

sys.path.append(Physics_Path);

import  logging;
from    copy            import  deepcopy;

import  torch;
import  numpy;

from    typing          import  TYPE_CHECKING;
if TYPE_CHECKING:
    from    Trainer     import  Trainer;
    from    Physics     import  Physics;


from    EncoderDecoder  import  EncoderDecoder;
from    MLP             import  act_dict;
from    Autoencoder     import  Autoencoder, load_Autoencoder;
# Set up logging.
LOGGER  : logging.Logger    = logging.getLogger(__name__);


# -------------------------------------------------------------------------------------------------
# Displacement, Velocity Autoencoder
# -------------------------------------------------------------------------------------------------

class Autoencoder_Pair(EncoderDecoder):
    """"
    This class defines a pair of auto-encoders for displacement, velocity data. Specifically, each 
    object consists of a pair of auto-encoders, one for processing displacement data and another 
    for processing velocity data. 
    """

    def __init__(   self,
                    Frame_Shape         : list[int],
                    config              : dict) -> None:
        """
        The initializer for the Autoencoder_Pair class. We assume that each input is a tuple 
        of data, (D, V), representing the displacement and velocity of some system at some point 
        in time. We encode D and V separately; each gets its own autoencoder with distinct weights. 


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        Frame_shape : list[int], len = k
            The shape of elements of the FOM space. This also specifies the final k dimensions of 
            the shape of the input to the first layer (if reshape_index == 0) or the output of 
            the last layer (if reshape_index == -1). 

        config: dict
            The "EncoderDecoder" sub dictionary of the configuration file. This must contain 
            a "type" key whose value is either "pair" or "autoencoder_pair". It must also contain 
            an item whose key matches the value of tye "type" key and whose value is a 
            sub-dictionary specifying the configuration settings for the autoencoder_pair object.
            Namely, it must contain the following sub-keys:
                - hidden_widths : A list of ints specifying the widths of the hidden layers in 
                each encoder
                - latent_dimension : An integer specifying the latent dimension of each autoencoder
                - activations : A list of strings specifying the activation functions for each 
                encoder. It's length must match that of hidden_widths, and each element must belong
                to the act_dict.
                - n_Decoders : an integer specifying the number of decoders we are to use for each 
                component of the solution.
            
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Input checks.
        assert 'type' in config;
        assert (config['type'] == "pair") or (config['type'] == "autoencoder_pair");
        pair_key  : str = config['type'];
        assert pair_key in config;
        pair_config             : dict              = config[pair_key];

        assert "hidden_widths"      in pair_config;
        assert "latent_dimension"   in pair_config;
        assert "activations"        in pair_config;
        assert "n_Decoders"         in pair_config;
        
        assert isinstance(Frame_Shape, list),                 "type(Frame_Shape) == %s, expected list" % (str(type(reshape_shape)));
        for i in range(len(Frame_Shape)):
            assert isinstance(Frame_Shape[i], int),           "type(Frame_Shape[%d]) = %s, expected int" % (i, str(type(reshape_shape[i])));
            assert Frame_Shape[i] > 0,                        "Frame_Shape[%d] = %d, needs to be positive" % (i, reshape_shape[i]);
        


        # Next, fetch the hidden widths and latent dimension (n_z). 
        hidden_widths           : list[int]         = pair_config['hidden_widths'];
        n_z                     : int               = pair_config['latent_dimension'];

        # Fetch the activations. This can either be a string or a list of strings. If it's 
        # a string, then we use that activation for all layers.
        n_hidden_layers     : int               = len(hidden_widths);
        if(isinstance(pair_config['activations'], str)):
            activations         : list[str]     = [pair_config['activations']] * n_hidden_layers;   # The final layer has no activation.
        elif(isinstance(pair_config['activations'], list)):
            activations         : list[str]     = pair_config['activations'];
            assert(len(activations) == n_hidden_layers);
        else:
            raise ValueError("Activations must be a string or a list of strings.");
        
        for i in range(len(activations)):
            assert isinstance(activations[i], str),             "type(activations[%d]) = %s, must be str" % (i, str(type(activations[i])));
            assert activations[i].lower() in act_dict.keys(),   "activations[%d] = %s; not in act_dict keys" % (i, activations[i].lower());
        

        # Now build the widths attribute + fetch Frame_Shape from physics.
        space_dim           : int               = numpy.prod(Frame_Shape).item();
        widths              : list[int]         = [space_dim] + hidden_widths + [n_z];
        for i in range(len(widths)):
            assert isinstance(widths[i], int),                  "type(widths[%d]) = %s, must be int" % (i, str(type(widths[i])));
            assert widths[i] > 0,                               "widths[%d] = %d, must be positive" % (i, widths[i]);
        assert numpy.prod(Frame_Shape) == widths[0],            "numpy.prod(self.reshape_shape) = %d, widths[0] = %d; must be equal" % (numpy.prod(reshape_shape), widths[0]);

        # Extract the number of decoders.
        n_Decoders = config[pair_key]['n_Decoders'];

        # Run the superclass initializer.
        super().__init__(n_IC = 2, n_z = widths[-1], n_Decoders = n_Decoders, config = config);
        LOGGER.info("Initializing an Autoencoder_Pair");

        # In general, the FOM solution may be vector valued and have multiple spatial dimensions. 
        # We need to know the shape of each FOM frame. 
        self.reshape_shape  : list[int]     = Frame_Shape; 
        
        # Fetch information about the domain/co-domain.
        self.widths         : list[int]     = widths;

        # Use the settings to set up the activation information for the encoder.
        self.activations    : list[str]     =  activations;

        # Make a config for the AE.
        ae_config = deepcopy(config);
        ae_config['ae'] = deepcopy(config[pair_key]);
        del ae_config[pair_key];

        # Next, build the velocity and displacement auto-encoders.
        LOGGER.info("Initializing the Displacement Autoencoder...");
        self.Displacement_Autoencoder   = Autoencoder(  widths          = widths, 
                                                        activations     = activations, 
                                                        reshape_shape   = self.reshape_shape,
                                                        config          = ae_config);

        LOGGER.info("Initializing the Velocity Autoencoder...");
        self.Velocity_Autoencoder       = Autoencoder(  widths          = widths, 
                                                        activations     = activations,
                                                        reshape_shape   = self.reshape_shape,
                                                        config          = ae_config);



    def Encode(self,
               Displacement_Frames  : torch.Tensor, 
               Velocity_Frames      : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This function encodes a set of displacement and velocity frames.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Displacement_Frames : torch.Tensor, shape = (N_Frames,) + self.reshape_shape
            Displacement_Frames[i, ...] represents the displacement portion of the i'th FOM frame.
            Here, N_Frames is the number of frames we want to encode and reshape_shape specifies 
            the shape of each frame. 

        Velocity_Frames : torch.Tensor, shape = (N_Frames,) + self.reshape_shape
            Velocity_Frames[i, ...] represents the velocity portion of the i'th FOM frame. Here, 
            N_Frames is the number of frames we want to encode for each parameter combination and 
            reshape_shape specifies the shape of each frame. 
        

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Latent_Displacement, Latent_Velocity
        
        Latent_Displacement : torch.Tensor, shape = (N_Frames, self.n_z))
            Latent_Displacement[i, :] represents the encoding of the displacement portion of the 
            i'th FOM frame

        Latent_Velocity : torch.Tensor, shape = (N_Frames, self.n_z))
            Latent_Velocity[i, :] represents the encoding of the velocity portion of the i'th FOM 
            frame.
        """

        # Check that we have the same number of displacement, velocity frames.
        assert isinstance(Displacement_Frames, torch.Tensor),   "type(Displacement_Frames) = %s; must be torch.Tensor" % str(type(Displacement_Frames));
        assert isinstance(Velocity_Frames, torch.Tensor),       "type(Velocity_Frames) = %s; must be torch.Tensor" % str(type(Velocity_Frames));
        assert len(Displacement_Frames.shape)       ==  len(self.reshape_shape) + 1,    "Displacement_Frames.shape = %s, length must be len(self.reshape_shape) (self.reshape_shape = %s) + 1" % (str(Displacement_Frames.shape), str(self.reshape_shape));
        assert Displacement_Frames.shape            ==  Velocity_Frames.shape,          "Displacement_Frames.shape = %s, Velocity_Frames.shape = %s" % (str(Displacement_Frames.shape), str(Velocity_Frames.shape));
        assert list(Displacement_Frames.shape[1:])  ==  self.reshape_shape,             "list(Displacement_Frames.shape[1:]) = %s, self.reshape_shape = %s; must be equal" % (str(list(Displacement_Frames.shape[1:])), str(self.reshape_shape));
    
        # Encode the displacement frames.
        Latent_Displacement : torch.Tensor = self.Displacement_Autoencoder.Encode( Displacement_Frames)[0];
        Latent_Velocity     : torch.Tensor = self.Velocity_Autoencoder.Encode(     Velocity_Frames)[0];

        # All done!
        return Latent_Displacement, Latent_Velocity;



    def Eval_Decoder(   self,
                        i_Decoder            : int, 
                        Latent_Displacement  : torch.Tensor, 
                        Latent_Velocity      : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This function decodes a set of displacement and velocity frames.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        i_Decoder : int
            the index of the decoder we want to use to decode Z.
        
        Latent_Displacement : torch.Tensor, shape = (N_Frames, self.n_z)
            i,j element represents the j'th component of the encoding of the displacement portion 
            of the i'th FOM frame.

        Latent_Velocity : torch.Tensor, shape = (N_Frames, self.n_z)
            i,j element represents the j'th component of the encoding of the velocity portion of 
            the i'th FOM frame.
     

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Reconstructed_Displacement, Reconstructed_Velocity
         
        Reconstructed_Displacement : torch.Tensor, shape = (N_Frames,) + self.reshape_shape
            Reconstructed_Displacement[i, ...] represents the reconstruction of the displacement 
            portion of the i'th FOM frame. 

        Reconstructed_Velocity : torch.Tensor, shape = (N_Frames,) + self.reshape_shape
            Reconstructed_Velocity[i, ...] represents the reconstruction of the velocity portion 
            of i'th FOM frame.
        """

        # Checks
        assert isinstance(Latent_Displacement, torch.Tensor),   "type(Latent_Displacement) = %s; must be torch.Tensor" % str(type(Latent_Displacement));
        assert isinstance(Latent_Velocity, torch.Tensor),       "type(Latent_Velocity) = %s; must be torch.Tensor" % str(type(Latent_Velocity));
        assert len(Latent_Displacement.shape)   == 2,           "Latent_Displacement.shape = %s; must have length 2" % str(Latent_Displacement.shape);
        assert Latent_Velocity.shape            == Latent_Displacement.shape,   "Latent_Velocity.shape = %s, Latent_Displacement.shape = %s; must be equal" % (str(Latent_Displacement.shape), str(Latent_Velocity.shape));
        assert (i_Decoder >= 0) and (i_Decoder < self.n_Decoders - 1),          "i_Decoder must be in {0, ... , %d}, got %d" % (self.n_Decoders - 1, i_Decoder);

        # Encode the displacement frames.
        Reconstructed_Displacement  : torch.Tensor  = self.Displacement_Autoencoder.Eval_Decoder( i_Decoder, Latent_Displacement)[0];
        Reconstructed_Velocity      : torch.Tensor  = self.Velocity_Autoencoder.Eval_Decoder( i_Decoder, Latent_Velocity)[0];

        # All done!
        return Reconstructed_Displacement, Reconstructed_Velocity;




    def export(self) -> dict:
        """
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        This function extracts everything we need to recreate self from scratch. Specifically, we 
        extract the encoder/decoder state dictionaries, self's architecture, activation function 
        and reshape_shape. We store and return this information in a dictionary.
         
        You can pass the returned dictionary to the load_Autoencoder_Pair method to generate an 
        Autoencoder object that is identical to self.
        """

        dict_ = {   'reshape_shape'     : self.reshape_shape,
                    'widths'            : self.widths,
                    'activations'       : self.activations,
                    'Displacement dict' : self.cpu().Displacement_Autoencoder.export(),
                    'Velocity dict'     : self.cpu().Velocity_Autoencoder.export(),
                    'config'            : self.config};
        return dict_;
    


def load_Autoencoder_Pair(dict_ : dict) -> Autoencoder_Pair:
    """
    This function builds a Autoencoder_Pair object using the information in dict_. dict_ should be 
    the dictionary returned by the export method for some Autoencoder_Pair object (or a 
    de-serialized version of one). The Autoencoder_Pair that we recreate should be an identical 
    copy of the object that generated dict_.


    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    dict_ : dict
        This should be a dictionary returned by a Autoencoder_Pair's export method.

    

    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    AEP : Autoencoder_Pair
        A Autoencoder_Pair object that is identical to the one that created dict_!
    """

    LOGGER.info("De-serializing an Autoencoder_Pair..." );

    # First, extract the information we need to initialize a Autoencoder_Pair object with the same 
    # architecture as the one that created dict_.
    reshape_shape   : list[int] = dict_['reshape_shape'];
    config          : dict      = dict_['config'];

    # Now initialize the Autoencoder_Pair.
    AEP                     = Autoencoder_Pair( reshape_shape   = reshape_shape,
                                                config          = config);
    
    # Now replace its auto-encoders.
    AEP.Displacement_Autoencoder    = load_Autoencoder(dict_['Displacement dict']);
    AEP.Velocity_Autoencoder        = load_Autoencoder(dict_['Velocity dict']);

    # All done!
    return AEP;