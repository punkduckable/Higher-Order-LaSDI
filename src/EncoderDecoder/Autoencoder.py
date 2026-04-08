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
                    reshape_shape   : list[int],
                    widths          : list[int], 
                    activations     : list[str]) -> None:
        r"""
        Initializes an Autoencoder object. An Autoencoder consists of two networks, an encoder, 
        E : \mathbb{R}^F -> \mathbb{R}^L, and a decoder, D : \mathbb{R}^L -> \marthbb{R}^F. We 
        assume that the dataset consists of samples of a parameterized L-manifold in 
        \mathbb{R}^F. The idea then is that E and D act like the inverse coordinate patch and 
        coordinate patch, respectively. In our case, E and D are trainable neural networks. We 
        try to train E and map data in \mathbb{R}^F to elements of a low dimensional latent 
        space (\mathbb{R}^L) which D can send back to the original data. (thus, E, and D should
        act like inverses of one another).


        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        reshape_shape : list[int]
            This is a list of k integers specifying the final k dimensions of the shape of the 
            input to the first layer (if reshape_index == 0) or the output of the last layer (if 
            reshape_index == -1). 
        
        widths : list[int]
            A list of integers specifying the widths of the layers in the encoder. We use the 
            revere of this list to specify the widths of the layers in the decoder. See the 
            docstring for the MultiLayerPerceptron class for details on how Widths defines a 
            network.

        activations : list[str], len = len(widths) - 2
            i'th element specifies which activation function we want to use after the i'th layer 
            in the encoder. The final layer has no activation function. We use the reversed list
            for the decoder. 



        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Checks
        assert isinstance(reshape_shape, list),             "type(reshape_shape) == %s, expected list" % (str(type(reshape_shape)));
        for i in range(len(reshape_shape)):
            assert isinstance(reshape_shape[i], int),           "type(reshape_shape[%d]) = %s, expected int" % (i, str(type(reshape_shape[i])));
            assert reshape_shape[i] > 0,                        "reshape_shape[%d] = %d, needs to be positive" % (i, reshape_shape[i]);
        assert isinstance(widths, list),                    "type(widths) = %s, expected list" % str(type(widths));
        for i in range(len(widths)):
            assert isinstance(widths[i], int),                  "type(widths[%d]) = %s, must be int" % (i, str(type(widths[i])));
            assert widths[i] > 0,                               "widths[%d] = %d, must be positive" % (i, widths[i]);
        assert isinstance(activations, list),               "type(activations) = %s, expected list" % str(type(activations));
        assert len(activations) == len(widths) - 2,         "len(activations) = %d, len(widths) = %d; but we must have len(activations) = len(widths) - 2" % (len(activations), len(widths));
        for i in range(len(activations)):
            assert isinstance(activations[i], str),             "type(activations[%d]) = %s, must be str" % (i, str(type(activations[i])));
            assert activations[i].lower() in act_dict.keys(),   "activations[%d] = %s; not in act_dict keys" % (i, activations[i].lower());
        assert numpy.prod(reshape_shape) == widths[0],      "numpy.prod(self.reshape_shape) = %d, widths[0] = %d; must be equal" % (numpy.prod(reshape_shape), widths[0]);

        # Run the superclass initializer.
        super().__init__(n_IC = 1, n_z = widths[-1]);
        
        # Store information (for return purposes).
        self.widths         : list[int] = widths;
        self.activations    : list[str] = activations;
        self.reshape_shape  : list[int] = reshape_shape;
        LOGGER.info("Initializing an Autoencoder with latent space dimension %d" % self.n_z);
        LOGGER.info("  Reshape shape: %s" % str(reshape_shape));
        LOGGER.info("  Widths: %s" % str(widths));
        LOGGER.info("  Activations: %s" % str(activations));


        # Build the encoder, decoder.
        LOGGER.info("Initializing the encoder...");
        self.encoder = MultiLayerPerceptron(
                            widths              = widths, 
                            activations         = activations,
                            reshape_index       = 0,                    # We need to flatten the spatial dimensions of each FOM frame.
                            reshape_shape       = reshape_shape);

        LOGGER.info("Initializing the decoder...");
        self.decoder = MultiLayerPerceptron(
                            widths              = widths[::-1],         # Reverses the order for the decoder.
                            activations         = activations[::-1],    # Reverses the order for the decoder.
                            reshape_index       = -1,                   # We need to reshape the network output to a FOM frame.
                            reshape_shape       = reshape_shape);       # We need to reshape the network output to a FOM frame.


        # All done!
        return;



    def Encode(self, U : torch.Tensor) -> tuple[torch.Tensor]:
        """
        This function encodes a set of displacement and velocity frames.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        X : torch.Tensor, shape = (n_Frames,) + self.reshape_shape
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
        assert len(U.shape)         ==  len(self.reshape_shape) + 1,    "U.shape = %s, self.reshape_shape = %s"     % (str(U.shape), str(self.reshape_shape));
        assert list(U.shape[1:])    ==  self.reshape_shape,             "U.shape[1:] = %s, self.reshape_shape = %s" % (str(U.shape[1:]), str(self.reshape_shape));
    
        # Encode the frames!
        return (self.encoder(U),);



    def Decode(self, Z : torch.Tensor) -> tuple[torch.Tensor]:
        """
        This function decodes a set of latent frames.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Z : torch.Tensor, shape = (n_Frames, self.n_z)
           i,j element holds the j'th component of the encoding of the i'th frame.
     

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        R : tuple[torch.Tensor], len = 1
            A single element tuple whose lone element is a torch.Tensor with shape = (n_Frames,) 
            + self.reshape_shape. R[i ...] represents the reconstruction of the i'th FOM frame.
        """

        # Check that the input has the correct shape. 
        assert len(Z.shape)   == 2, "Z.shape = %s, must have length 2." % str(Z.shape);
    
        # Decode the frames!
        # NOTE: Return a tuple for consistency with the EncoderDecoder interface and with
        # Autoencoder.Encode(...), which returns a 1-tuple.
        return (self.decoder(Z),);



    def forward(self, X : torch.Tensor) -> tuple[torch.Tensor]:
        """
        This function passes X through the encoder, producing a latent state, Z. It then passes 
        Z through the decoder; hopefully producing a vector that approximates X.
        

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        U : torch.Tensor, shape = (n_Frames,) + self.reshape_shape
            A tensor holding a batch of inputs. We pass this tensor through the encoder + decoder 
            and then return the result.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Y : tuple[torch.Tensor], len = 1
            A single element tuple whose lone element is a torch.Tensor of shape = X.shape holding 
            the image of X under the encoder and decoder. 
        """

        # Encoder the input
        Z : torch.Tensor            = self.Encode(X)[0];

        # Now decode z.
        Y : tuple[torch.Tensor]     = self.Decode(Z);

        # All done! Hopefully Y \approx X.
        return Y;



    def export(self) -> dict:
        """
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        This function extracts everything we need to recreate self from scratch. Specifically, we 
        extract the encoder/decoder state dictionaries, self's architecture, activation function 
        and reshape_shape. We store and return this information in a dictionary.
         
        You can pass the returned dictionary to the load_Autoencoder method to generate an 
        Autoencoder object that is identical to self.
        """

        # TO DO: deep export which includes all information needed to re-initialize self from 
        # scratch. This would probably require changing the initializer.

        dict_ = {   'encoder state'  : self.encoder.cpu().state_dict(),
                    'decoder state'  : self.decoder.cpu().state_dict(),
                    'widths'         : self.widths, 
                    'activations'    : self.activations, 
                    'reshape_shape'  : self.reshape_shape};
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
    widths          : list[int] = dict_['widths'];
    activations     : list[str] = dict_['activations'];
    reshape_shape   : list[int] = dict_['reshape_shape'];

    # Now... initialize an Autoencoder object.
    AE = Autoencoder(widths = widths, activations = activations, reshape_shape = reshape_shape);

    # Now, update the encoder/decoder parameters.
    AE.encoder.load_state_dict(dict_['encoder state']); 
    AE.decoder.load_state_dict(dict_['decoder state']); 

    # All done, AE is now identical to the Autoencoder object that created dict_.
    return AE;
