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
                    reshape_shape       : list[int],
                    widths              : list[int],
                    activations         : list[str]) -> None:
        """
        The initializer for the Autoencoder_Pair class. We assume that each input is a tuple 
        of data, (D, V), representing the displacement and velocity of some system at some point 
        in time. We encode D and V separately; each gets its own autoencoder with distinct weights. 


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        reshape_shape : list[int], len = k
            specifies the final k dimensions of the shape of the input to the first layer (if 
            reshape_index == 0) or the output of the last layer (if reshape_index == -1). 

        widths : list[int]
            specifies the widths of the layers in each encoder. See Autoencoder docstring.

        activations : list[str], len = len(widths) - 2
            i'th element specifies which activation function we want to use after the i'th layer 
            in each encoder (decoder gets the reverse order). The final layer has no activation 
            function. 
    
            
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

        # Call the super class initializer.
        super().__init__(n_IC = 2, n_z = widths[-1]);
        LOGGER.info("Initializing an Autoencoder_Pair");

        # In general, the FOM solution may be vector valued and have multiple spatial dimensions. 
        # We need to know the shape of each FOM frame. 
        self.reshape_shape  : list[int]     = reshape_shape; 
        
        # Fetch information about the domain/co-domain.
        self.widths         : list[int]     = widths;

        # Use the settings to set up the activation information for the encoder.
        self.activations    : list[str]     =  activations;

        # Next, build the velocity and displacement auto-encoders.
        LOGGER.info("Initializing the Displacement Autoencoder...");
        self.Displacement_Autoencoder   = Autoencoder(  widths          = widths, 
                                                        activations     = activations, 
                                                        reshape_shape   = self.reshape_shape);

        LOGGER.info("Initializing the Velocity Autoencoder...");
        self.Velocity_Autoencoder       = Autoencoder(  widths          = widths, 
                                                        activations     = activations,
                                                        reshape_shape   = self.reshape_shape);



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



    def Decode(self,
               Latent_Displacement  : torch.Tensor, 
               Latent_Velocity      : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This function decodes a set of displacement and velocity frames.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

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

        # Check that we have the same number of displacement, velocity frames.
        assert isinstance(Latent_Displacement, torch.Tensor),   "type(Latent_Displacement) = %s; must be torch.Tensor" % str(type(Latent_Displacement));
        assert isinstance(Latent_Velocity, torch.Tensor),       "type(Latent_Velocity) = %s; must be torch.Tensor" % str(type(Latent_Velocity));
        assert len(Latent_Displacement.shape)   == 2,           "Latent_Displacement.shape = %s; must have length 2" % str(Latent_Displacement.shape);
        assert Latent_Velocity.shape            == Latent_Displacement.shape,   "Latent_Velocity.shape = %s, Latent_Displacement.shape = %s; must be equal" % (str(Latent_Displacement.shape), str(Latent_Velocity.shape));

        # Encode the displacement frames.
        Reconstructed_Displacement  : torch.Tensor  = self.Displacement_Autoencoder.Decode( Latent_Displacement)[0];
        Reconstructed_Velocity      : torch.Tensor  = self.Velocity_Autoencoder.Decode(     Latent_Velocity)[0];

        # All done!
        return Reconstructed_Displacement, Reconstructed_Velocity;



    def forward(self, Displacement_Frames : torch.Tensor, Velocity_Frames : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The forward method for the Autoencoder_Pair class. It encodes and then decodes 
        Displacement_Frames and Velocity_Frames.



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

        Reconstructed_Displacement, Reconstructed_Velocity

        Reconstructed_Displacement : torch.Tensor, shape = (N_Frames,) + self.reshape_shape
            Reconstructed_Displacement[i, ...] represents the reconstruction of the displacement 
            portion of the i'th FOM frame. 

        Reconstructed_Velocity : torch.Tensor, shape = (N_Frames,) + self.reshape_shape
            Reconstructed_Velocity[i, ...] represents the reconstruction of the velocity portion 
            of i'th FOM frame.
        """

        # Encode the displacement, velocity frames
        Latent_Displacement, Latent_Velocity = self.Encode(     Displacement_Frames   = Displacement_Frames, 
                                                                Velocity_Frames       = Velocity_Frames);

        # Now reconstruct displacement, velocity.
        Reconstructed_Displacement, Reconstructed_Velocity = self.Decode(
                                                                Latent_Displacement = Latent_Displacement, 
                                                                Latent_Velocity     = Latent_Velocity);

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
                    'Velocity dict'     : self.cpu().Velocity_Autoencoder.export()};
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
    widths          : list[int] = dict_['widths'];
    activations     : list[str] = dict_['activations'];

    # Now initialize the Autoencoder_Pair.
    AEP                     = Autoencoder_Pair( widths          = widths, 
                                                activations     = activations,
                                                reshape_shape   = reshape_shape);
    
    # Now replace its auto-encoders.
    AEP.Displacement_Autoencoder    = load_Autoencoder(dict_['Displacement dict']);
    AEP.Velocity_Autoencoder        = load_Autoencoder(dict_['Velocity dict']);

    # All done!
    return AEP;