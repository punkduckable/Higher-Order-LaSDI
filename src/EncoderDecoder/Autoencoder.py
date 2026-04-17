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

import  torch;
import  numpy;

from    typing          import  TYPE_CHECKING;
if TYPE_CHECKING:
    from    Trainer     import  Trainer;
    from    Physics     import  Physics;


from    EncoderDecoder  import  EncoderDecoder;
from    MLP             import  MultiLayerPerceptron, act_dict;
# Set up logging.
LOGGER  : logging.Logger    = logging.getLogger(__name__);




# -------------------------------------------------------------------------------------------------
# Autoencoder class
# -------------------------------------------------------------------------------------------------

class Autoencoder(EncoderDecoder):
    def __init__(   self,         
                    Frame_Shape     : list[int],    
                    config          : dict) -> None:
        r"""
        Initializes an Autoencoder object. An Autoencoder consists of two networks, an encoder, 
        E : \mathbb{R}^F -> \mathbb{R}^L, and a Decoder, D : \mathbb{R}^L -> \marthbb{R}^F. 

        We assume that the dataset consists of samples of a parameterized L-manifold in 
        \mathbb{R}^F. The idea then is that E and D act like the inverse coordinate patch and 
        coordinate patch, respectively. In our case, E is a neural network, and D is a linear 
        combination of sub-decoders (to allow for multi-stage training), each one of which is a 
        Neural Network.
        
        We try to train E and map data in \mathbb{R}^F to elements of a low dimensional latent 
        space (\mathbb{R}^L) which D can send back to the original data. (thus, E, and D should
        act like inverses of one another).


        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        Frame_Shape : list[int]
            The shape of elements of the FOM space. This also specifies the final k dimensions of 
            the shape of the input to the first layer (if reshape_index == 0) or the output of 
            the last layer (if reshape_index == -1). 
        
        config: dict
            The "EncoderDecoder" sub dictionary of the configuration file. This must contain 
            a "type" key whose value is either "ae" or "autoencoder". It must also contain 
            an item whose key matches the value of tye "type" key and whose value is a 
            sub-dictionary specifying the configuration settings for the autoencoder object.
            Namely, it must contain the following sub-keys:
                - hidden_widths : A list of ints specifying the widths of the hidden layers
                - latent_dimension : An integer specifying the latent dimension
                - activations : A list of strings specifying the activation functions. It's 
                length must match that of hidden_widths, and each element must belong to the act_dict.
                - n_Decoders : an integer specifying the number of decoders we are to use.




        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """
        
        # Input checks.
        assert 'type' in config;
        assert (config['type'] == "ae") or (config['type'] == "autoencoder");
        ae_key  : str = config['type'];
        assert ae_key in config;
        ae_config               : dict              = config[ae_key];

        assert "hidden_widths"      in ae_config;
        assert "latent_dimension"   in ae_config;
        assert "activations"        in ae_config;
        assert "n_Decoders"         in ae_config;
        
        assert isinstance(Frame_Shape, list),                 "type(Frame_Shape) == %s, expected list" % (str(type(Frame_Shape)));
        for i in range(len(Frame_Shape)):
            assert isinstance(Frame_Shape[i], int),           "type(Frame_Shape[%d]) = %s, expected int" % (i, str(type(Frame_Shape[i])));
            assert Frame_Shape[i] > 0,                        "Frame_Shape[%d] = %d, needs to be positive" % (i, Frame_Shape[i]);
        


        # Next, fetch the hidden widths and latent dimension (n_z). 
        hidden_widths           : list[int]         = ae_config['hidden_widths'];
        n_z                     : int               = ae_config['latent_dimension'];

        # Fetch the activations. This can either be a string or a list of strings. If it's 
        # a string, then we use that activation for all layers.
        n_hidden_layers     : int               = len(hidden_widths);
        if(isinstance(ae_config['activations'], str)):
            activations         : list[str]     = [ae_config['activations']] * n_hidden_layers;   # The final layer has no activation.
        elif(isinstance(ae_config['activations'], list)):
            activations         : list[str]     = ae_config['activations'];
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
        assert numpy.prod(Frame_Shape) == widths[0],            "numpy.prod(self.Frame_Shape) = %d, widths[0] = %d; must be equal" % (numpy.prod(Frame_Shape), widths[0]);

        # Extract the number of decoders.
        n_Decoders = config[ae_key]['n_Decoders'];

        # Run the superclass initializer.
        super().__init__(n_IC = 1, n_z = widths[-1], n_Decoders = n_Decoders, config = config);
        
        # Store information (for return purposes).
        self.widths         : list[int] = widths;
        self.activations    : list[str] = activations;
        self.Frame_Shape    : list[int] = Frame_Shape;
        LOGGER.info("Initializing an Autoencoder with latent space dimension %d" % self.n_z);
        LOGGER.info("  Reshape shape: %s" % str(Frame_Shape));
        LOGGER.info("  Widths: %s" % str(widths));
        LOGGER.info("  Activations: %s" % str(activations));


        # Build the encoder, decoder.
        LOGGER.info("Initializing the encoder...");
        self.encoder = MultiLayerPerceptron(
                            widths              = widths, 
                            activations         = activations,
                            reshape_index       = 0,                    # We need to flatten the spatial dimensions of each FOM frame.
                            reshape_shape       = Frame_Shape);

        LOGGER.info("Initializing the decoder...");
        self.decoders = torch.nn.ModuleList([]);
        for i in range(n_Decoders):
            self.decoders.append(MultiLayerPerceptron(
                                    widths              = widths[::-1],         # Reverses the order for the decoder.
                                    activations         = activations[::-1],    # Reverses the order for the decoder.
                                    reshape_index       = -1,                   # We need to reshape the network output to a FOM frame.
                                    reshape_shape       = Frame_Shape));      # We need to reshape the network output to a FOM frame.

        # All done!
        return;



    def Encode(self, U : torch.Tensor) -> tuple[torch.Tensor]:
        """
        This function encodes a set of displacement and velocity frames.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        X : torch.Tensor, shape = (n_Frames,) + self.Frame_Shape
            X[i, ...] holds the i'th frame we want to encode. 


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Z : tuple[torch.Tensor], len = 1 
            single element is a torch.Tensor of shape = (n_Frasmes, self.n_z) whose i,j element 
            holds the j'th component of the encoding of the i'th FOM frame.
        """

        # Check that the inputs have the correct shape.
        assert isinstance(U, torch.Tensor),                             "type(U) = %s, must be torch.Tensor" % type(U);
        assert len(U.shape)         ==  len(self.Frame_Shape) + 1,    "U.shape = %s, self.Frame_Shape = %s"     % (str(U.shape), str(self.Frame_Shape));
        assert list(U.shape[1:])    ==  self.Frame_Shape,             "U.shape[1:] = %s, self.Frame_Shape = %s" % (str(U.shape[1:]), str(self.Frame_Shape));
    
        # Encode the frames!
        return (self.encoder(U),);



    def Eval_Decoder(self, i_Decoder : int, Z : torch.Tensor) -> tuple[torch.Tensor]:
        """
        This function decodes a set of latent frames.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        i_Decoder : int
            the index of the decoder we want to use to decode Z.
        
        Z : torch.Tensor, shape = (n_Frames, self.n_z)
           i,j element holds the j'th component of the encoding of the i'th frame.
     

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        R : tuple[torch.Tensor], len = 1
            A single element tuple whose lone element is a torch.Tensor with shape = (n_Frames,) 
            + self.Frame_Shape. R[i ...] represents the reconstruction of the i'th FOM frame.
        """

        # Checks 
        assert len(Z.shape)   == 2,                                     "Z.shape = %s, must have length 2." % str(Z.shape);
        assert (i_Decoder >= 0) and (i_Decoder < self.n_Decoders),      "i_Decoder must be in {0, ... , %d}, got %d" % (self.n_Decoders - 1, i_Decoder);

        # Decode the frames!
        # NOTE: Return a tuple for consistency with the EncoderDecoder interface and with
        # Autoencoder.Encode(...), which returns a 1-tuple.
        return (self.decoders[i_Decoder](Z),);



    def export(self) -> dict:
        """
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        This function extracts everything we need to recreate self from scratch. Specifically, we 
        extract the encoder/decoder state dictionaries, self's architecture, activation function 
        and Frame_Shape. We store and return this information in a dictionary.
         
        You can pass the returned dictionary to the load_Autoencoder method to generate an 
        Autoencoder object that is identical to self.
        """

        # TO DO: deep export which includes all information needed to re-initialize self from 
        # scratch. This would probably require changing the initializer.

        decoder_states_list = [];
        for i in range(self.n_Decoders):
            decoder_states_list.append(self.decoders[i].cpu().state_dict());

        dict_ = {   'EncoderDecoder dict'   : super().export(),
                    'encoder_state'         : self.encoder.cpu().state_dict(),
                    'decoder_states'        : decoder_states_list,
                    'widths'                : self.widths, 
                    'activations'           : self.activations, 
                    'Frame_Shape'           : self.Frame_Shape,
                    'config'                : self.config};
        return dict_;



def load_Autoencoder(dict_ : dict) -> Autoencoder:
    """
    This function builds an Autoencoder object using the information in dict_. dict_ should be 
    the dictionary returned by the export method for some Autoencoder object (or a de-serialized 
    version of one). The Autoencoder that we recreate should be an identical copy of the object 
    that generated dict_.

    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    dict_: dict
        This should be a dictionary returned by an Autoencoder's export method.

    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    AE : Autoencoder 
        An Autoencoder object that is identical to the one that created dict_!
    """

    LOGGER.info("De-serializing an Autoencoder..." );

    # First, extract the parameters we need to initialize an Autoencoder object with the same 
    # architecture as the one that created dict_.
    Frame_Shape     : list[int] = dict_['Frame_Shape'];
    config          : dict      = dict_['config'];

    # Now... initialize an Autoencoder object.
    AE = Autoencoder(Frame_Shape = Frame_Shape, config = config);

    # Set the Decoder_Weights, Active.
    AE.load(dict_ = dict_['EncoderDecoder dict']);

    # Now, update the encoder/decoder parameters.
    AE.encoder.load_state_dict(dict_['encoder_state']); 

    decoder_states  : list  = dict_['decoder_states'];
    n_Decoders      : int   = len(decoder_states);
    for i in range(n_Decoders):
        AE.decoders[i].load_state_dict(decoder_states[i]); 

    # All done, AE is now identical to the Autoencoder object that created dict_.
    return AE;
