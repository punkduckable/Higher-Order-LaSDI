# -------------------------------------------------------------------------------------------------
# Import and Setup
# -------------------------------------------------------------------------------------------------


# Add the Physics directory to the search path.
import  sys;
import  os;
src_Path        : str   = os.path.abspath(os.path.dirname(os.path.dirname(__file__)));
Physics_Path    : str  = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, "Physics"));
sys.path.append(Physics_Path);

import  logging;

import  torch;
import  numpy;

from    typing          import  TYPE_CHECKING;
if TYPE_CHECKING:
    from    Trainer     import  Trainer;
    from    Physics     import  Physics;


# Set up logging.
LOGGER  : logging.Logger    = logging.getLogger(__name__);




# -------------------------------------------------------------------------------------------------
# EncoderDecoder
# -------------------------------------------------------------------------------------------------

class EncoderDecoder(torch.nn.Module):
    def __init__(   self, 
                    n_IC    : int, 
                    n_z     : int) -> None:
        r"""
        Initializes a EncoderDecoder object. A EncoderDecoder object does two things. a) It can 
        encode FOM states (frames) to their latent encodings, and b) it can decode those latent
        encodings back to the FOM state. An EncoderDecoder object is defined by two variables:

            n_IC (the number of initial conditions)
            n_z (the latent space dimension)
        
        The encoder must map n_IC elements of the FOM space to n_IC elements of \\mathbb{R}^{n_z}. 
        The decoder must map n_IC elements of \\mathbb{R}^{n_z} to n_IC elements of the FOM space.

        To implement a EncoderDecoder subclass, you must implement the Encode, Decode, forward, 
        and latent_initial_conditions, and save methods. You should also implement a "load" 
        function to load the encoded state from a save.


        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        n_IC : int
            The number of initial conditions. The encoder must accept this many elements of the 
            FOM space and map them to the same number of elements of the latent space.

        n_z : int 
            The latent space dimension.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Checks
        assert n_IC > 0,    "n_IC = %d; must be positive" % n_IC;
        assert n_z > 0,     "n_z = %d; must be positive" % n_z;

        # Run the superclass initializer.
        super().__init__();
        
        # Store information (for return purposes).
        self.n_IC           : int       = n_IC;
        self.n_z            : int       = n_z;

        # All done!
        return;



    def Encode(self, X : tuple[torch.Tensor]) -> tuple[torch.Tensor]:
        """
        This function should accept n_IC elements of the FOM space and map them to n_IC elements 
        of the latent space. The output must be a tuple of tensors. The input can either be n_IC 
        distinct variables, or one tuple of length n_IC.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        X : tuple[torch.Tensor], len = self.n_IC
            The inputs to be encoded


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Z : tuple[torch.Tensor], len = self.n_IC
            A List of n_IC elements of \\mathbb{R}^{n_z}. 
        """

        raise RuntimeError("Abstract method EncoderDecoder.Encode!");



    def Decode(self, Z : tuple[torch.Tensor]) -> tuple[torch.Tensor]:
        """
        This function should accept n_IC elements of the latent space and map them to n_IC elements 
        of the FOM space. The output must be a tuple of tensors. The input can either be n_IC 
        distinct variables, or one tuple of length n_IC.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Z : tuple[torch.Tensor], len = self.n_IC
            The latent states to be decoded.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        X : tuple[torch.Tensor], len = self.n_IC
            A List of n_IC elements of the FOM space 
        """

        raise RuntimeError("Abstract method EncoderDecoder.Decode!");



    def forward(self, X : tuple[torch.Tensor]) -> tuple[torch.Tensor]:
        """
        This function passes X through the encoder, producing a latent state, Z. It then passes 
        Z through the decoder; hopefully producing a set of vectors that approximates X.
        

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        X : tuple[torch.Tensor], len = self.n_IC
            The inputs to the encoder.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Y : tuple[torch.Tensor], len = self.n_IC
            A self.n_IC element tuple of torch.Tensors in the FOM space holding the image of X 
            under the encoder and decoder. 
        """

        raise RuntimeError("Abstract method EncoderDecoder.forward!");



    def latent_initial_conditions(  self,
                                    param_grid     : numpy.ndarray, 
                                    physics        : "Physics",
                                    trainer        : "Trainer") -> list[list[numpy.ndarray]]:
        """
        This function maps a set of initial conditions for the FOM to initial conditions for the 
        latent space dynamics. Specifically, we take in a set of possible parameter values. For 
        each set of parameter values, we recover the FOM IC (from physics), then map this FOM IC to 
        a latent space IC (by encoding it). We do this for each parameter combination and then 
        return a list housing the latent space ICs.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param_grid : numpy.ndarray, shape = (n_param, n_p)
            i,j element of this array holds the value of the j'th parameter in the i'th combination of 
            parameters. Here, n_param is the number of combinations of parameter values and n_p is the 
            number of parameters (in each combination).

        physics : "Physics"
            A "Physics" object that, among other things, stores the IC for each combination of 
            parameter values. This physics object should have the same number of initial conditions as 
            self.
        
        trainer : "Trainer"
            The trainer object used to train the EncoderDecoder.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        Z0 : list[list[numpy.ndarray]], len = n_param
            An n_param element list whose i'th element is an n_IC element list whose j'th element 
            is an numpy.ndarray of shape (1, n_z) whose k'th element holds the k'th component of 
            the encoding of the initial condition for the j'th derivative of the latent dynamics 
            corresponding to the i'th combination of parameter values.
        
            If we let U0_i denote the FOM IC for the i'th set of parameters, then the i'th element of 
            the returned list is [self.encoder(U0_i)].
        """

        raise RuntimeError("Abstract method EncoderDecoder.latent_initial_conditions!");



    def export(self) -> dict:
        """
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        This function extracts everything we need to recreate self from scratch.
        """

        raise RuntimeError("Abstract method EncoderDecoder.export!");


